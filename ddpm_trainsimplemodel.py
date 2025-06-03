import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from tqdm import tqdm
import random

from ddpm_trainsimplemodel import SelfAttention, CrossAttention
from ddpm import DDPM

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


num_channels = 64
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=4*num_channels):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        # Нормализация + Self-Attention
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Нормализация + FFN с GeGLU
        residue_short = x
        x = self.layernorm_2(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(3, num_channels, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(num_channels, num_channels), UNET_AttentionBlock(8, num_channels//8)),
            SwitchSequential(UNET_ResidualBlock(num_channels, num_channels), UNET_AttentionBlock(8, num_channels//8)),
            SwitchSequential(nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(num_channels, 2*num_channels), UNET_AttentionBlock(8, num_channels//4)),
            SwitchSequential(UNET_ResidualBlock(2*num_channels, 2*num_channels), UNET_AttentionBlock(8, num_channels//4)),
            SwitchSequential(nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(2*num_channels, 4*num_channels), UNET_AttentionBlock(8, num_channels//2)),
            SwitchSequential(UNET_ResidualBlock(4*num_channels, 4*num_channels), UNET_AttentionBlock(8, num_channels//2)),
            SwitchSequential(nn.Conv2d(4*num_channels, 4*num_channels, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(4*num_channels, 4*num_channels)),
            SwitchSequential(UNET_ResidualBlock(4*num_channels, 4*num_channels)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(4*num_channels, 4*num_channels),
            UNET_AttentionBlock(8, num_channels//2),
            UNET_ResidualBlock(4*num_channels, 4*num_channels),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(8*num_channels, 4*num_channels)),
            SwitchSequential(UNET_ResidualBlock(8*num_channels, 4*num_channels)),
            SwitchSequential(UNET_ResidualBlock(8*num_channels, 4*num_channels), Upsample(4*num_channels)),
            SwitchSequential(UNET_ResidualBlock(8*num_channels, 4*num_channels), UNET_AttentionBlock(8, num_channels//2)),
            SwitchSequential(UNET_ResidualBlock(8*num_channels, 4*num_channels), UNET_AttentionBlock(8, num_channels//2)),
            SwitchSequential(UNET_ResidualBlock(6*num_channels, 4*num_channels), UNET_AttentionBlock(8, num_channels//2), Upsample(4*num_channels)),
            SwitchSequential(UNET_ResidualBlock(6*num_channels, 2*num_channels), UNET_AttentionBlock(8, num_channels//4)),
            SwitchSequential(UNET_ResidualBlock(4*num_channels, 2*num_channels), UNET_AttentionBlock(8, num_channels//4)),
            SwitchSequential(UNET_ResidualBlock(3*num_channels, 2*num_channels), UNET_AttentionBlock(8, num_channels//4), Upsample(2*num_channels)),
            SwitchSequential(UNET_ResidualBlock(3*num_channels, num_channels), UNET_AttentionBlock(8, num_channels//8)),
            SwitchSequential(UNET_ResidualBlock(2*num_channels, num_channels), UNET_AttentionBlock(8, num_channels//8)),
            SwitchSequential(UNET_ResidualBlock(2*num_channels, num_channels), UNET_AttentionBlock(8, num_channels//8)),
        ])

    def forward(self, x, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, time)
            skip_connections.append(x)

        x = self.bottleneck(x, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, time)

        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(num_channels)
        self.unet = UNET()
        self.final = UNET_OutputLayer(num_channels, 3)

    def forward(self, latent, time):
        time = self.time_embedding(time)
        output = self.unet(latent, time)
        output = self.final(output)
        return output

WIDTH = 64
HEIGHT = 64

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def read_dataset(image_folder, batch_size=1, num_epochs=1, shuffle=True, device="cuda"):
    image_names = [entry.name for entry in os.scandir(image_folder) if entry.is_file()][:200]

    # Собираем все эмбединги
    all_images = []

    # Обрабатываем каждое изображение с его описаниями
    processed_images = set()
    print("Преобразование изображений")
    timesteps = tqdm(image_names)
    for image_name in timesteps:
        image_path = os.path.join(image_folder, image_name)

        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            continue

        # Открываем изображение только один раз, даже если несколько описаний
        if image_name not in processed_images:
            try:
                input_image = Image.open(image_path)
                input_image_tensor = input_image.resize((WIDTH, HEIGHT))
                input_image_tensor = np.array(input_image_tensor)
                input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
                input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
                input_image_tensor = input_image_tensor.unsqueeze(0)
                input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
                processed_images.add(image_name)
                all_images.append(input_image_tensor)
            except:
                continue


    # Преобразуем списки в тензоры
    image_embeddings_tensor = torch.cat(all_images, dim=0)

    # Создаем Dataset и DataLoader
    class EmbeddingsDataset(Dataset):
        def __init__(self, images):
            self.images = images

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx]

    dataset = EmbeddingsDataset(image_embeddings_tensor)
    return dataset



diffusion = Diffusion().to("cuda")
diffusion.to("cuda")
def train(dataset, batch_size=1, num_epochs=100, shuffle=True, device="cuda"):
    if device:
        to_idle = lambda x: x.to(device)
    else:
        to_idle = lambda x: x

    generator = torch.Generator(device=device)
    generator.seed()

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()
    scheduler = DDPM(generator)

    print("\nЭпоха обучения:")
    timesteps = tqdm([i for i in range(num_epochs)])
    for epoch in timesteps:
        diffusion.train()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=False
        )
        avg_loss = 0
        for data in dataloader:
            latents = data

            # Добавление шума к latents
            timestep = random.choice(scheduler.timesteps)
            latents_noises, original_noises = scheduler.add_noise_training(latents, timestep)
            time_embedding = get_time_embedding(timestep).to(device)
            optimizer.zero_grad()
            # model_output — прогнозируемый шум
            model_output = diffusion(latents_noises, time_embedding)
            loss = mse(model_output, original_noises)
            avg_loss += loss
            loss.backward()
            optimizer.step()
            to_idle(diffusion)
        avg_loss = avg_loss / len(dataloader)
        print(f"\nLoss: {avg_loss}\n")
    torch.save(
        diffusion.state_dict(),
        "trained_diffusion_model.pth",
    )

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=num_channels//2, dtype=torch.float32) / num_channels//2)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



WIDTH = 64
HEIGHT = 64
def generate(
    n_inference_steps=50,
    seed=42,
    batch_size=1,
    device="cuda"
):
    with torch.no_grad():
        if device:
            to_idle = lambda x: x.to(device)
        else:
            to_idle = lambda x: x

        # Инициализация генератора случайных чисел в соответствии с указанным начальным числом
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)


        scheduler = DDPM(generator)
        scheduler.set_inference_timesteps(n_inference_steps)

        latents_shape = (1, 3, HEIGHT, WIDTH)

        latents = torch.randn(latents_shape, generator=generator, device=device)

        timesteps = tqdm(scheduler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents


            model_input = model_input.repeat(batch_size, 1, 1, 1)

            # model_output — прогнозируемый шум
            model_output = diffusion(model_input, time_embedding)
            latents = scheduler.step(timestep, latents, model_output)

        to_idle(diffusion)

        images = latents

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def generate_noise(
    seed=42,
    device="cuda"
):
        # Инициализация генератора случайных чисел в соответствии с указанным начальным числом
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        latents_shape = (1, 3, HEIGHT, WIDTH)
        latents = torch.randn(latents_shape, generator=generator, device=device)

        images = latents

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
