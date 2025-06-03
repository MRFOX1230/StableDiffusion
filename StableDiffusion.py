from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from generator import generate

# Text2Image
# Пример
prompt = "Собака в очках"
negative_prompt = ""
num_inference_steps = 50
seed = 42

output_image = generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    n_inference_steps=num_inference_steps,
    seed=seed
)

# Вывод получаемого изображения
Image.fromarray(output_image)




# # Image2Image
# # Пример
# prompt = "Сова играет на пианино"
# negative_prompt = ""
# num_inference_steps = 50
# seed = 42
# path = "путь к папке, где находится папка StableDiffusion"
# image_path = path + "/StableDiffusion/owl.jpg"
# strength = 0.8 # Влияет на то, как сильно будет зашумляться исходное изображение для дальнейшего восстановления по тексту

# input_image = Image.open(image_path)
# output_image = generate(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     input_image=input_image,
#     strength=strength,
#     n_inference_steps=num_inference_steps,
#     seed=seed
# )

# # Вывод получаемого изображения
# Image.fromarray(output_image)