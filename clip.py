import torch
from torch import nn
from torch.nn import functional as F
from attentions import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # Обучаемая матрица весов кодирует информацию о позиции каждого токена
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        # Предобработка нормализации перед вниманием
        self.layernorm_1 = nn.LayerNorm(n_embd)

        self.attention = SelfAttention(n_head, n_embd)
        # Предобработка нормализации перед FNN
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Полносвязный слой
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x

        # Self-Attention

        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # Применяем полносвязный слой, где скрытая размерность в 4 раза больше размерности вложения.
        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)

        # Применяем слои энкодера, аналогичные энкодеру Transformer
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)

        return output