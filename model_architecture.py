import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity=0.7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity = sparsity
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Dynamic sparsity
        if self.sparsity > 0:
            k = attn.shape[-1]
            num_keep = int(k * (1 - self.sparsity))
            values, _ = torch.topk(attn, num_keep, dim=-1)
            threshold = values[:, :, :, -1].unsqueeze(-1)
            mask = (attn >= threshold).float()
            attn = attn * mask
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, sparsity=0.7):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DynamicSparseAttention(embed_dim, num_heads, sparsity)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LightweightSparseTransformer(nn.Module):
    def __init__(self, input_dim=257, embed_dim=256, num_heads=8, depth=4, sparsity=0.7):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, sparsity=sparsity)
            for _ in range(depth)
        ])
        self.decoder = nn.Linear(embed_dim, input_dim)
        
    def forward(self, x):
        # x: (B, F, T)
        x = x.permute(0, 2, 1)  # (B, T, F)
        x = self.embed(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.decoder(x)
        return x.permute(0, 2, 1)