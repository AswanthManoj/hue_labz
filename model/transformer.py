import math
import json
import torch
import colorsys
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.block(x) + x)

class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.attn(x1, x1, x1)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class AutoEncoderColorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18
        transformer_dim = 64  # Reduced dimension for stability

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6 * transformer_dim),  # Direct output to token dimension
            nn.LayerNorm(6 * transformer_dim)
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            SimpleTransformerBlock(transformer_dim, heads=4, dropout=0.1)
            for _ in range(2)
        ])

        # Theme-specific processors
        self.light_processor = nn.Sequential(
            nn.Linear(6 * transformer_dim + self.input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(512),
            nn.Linear(512, 162),
            nn.Sigmoid()
        )

        self.dark_processor = nn.Sequential(
            nn.Linear(6 * transformer_dim + self.input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(512),
            nn.Linear(512, 162),
            nn.Sigmoid()
        )

        # Algorithmic parameters
        self.dark_bg_level = nn.Parameter(torch.ones(1) * 0.12)
        self.light_text_level = nn.Parameter(torch.ones(1) * 0.85)
        self.accent_tint_strength = nn.Parameter(torch.ones(1) * 0.05)
        self.algo_neural_balance = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, 6, 3)

        # Create algorithmic bases
        light_base = self._create_light_base(x_reshaped)
        dark_base = self._create_dark_base(x_reshaped)

        # Encode input to token space
        tokens = self.encoder(x).view(batch_size, 6, -1)  # [batch, 6, transformer_dim]

        # Process with transformer
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Flatten tokens and combine with input
        tokens_flat = tokens.reshape(batch_size, -1)
        combined_features = torch.cat([tokens_flat, x], dim=1)

        # Generate themes
        light_output = self.light_processor(combined_features).view(batch_size, 6, 9, 3)
        dark_output = self.dark_processor(combined_features).view(batch_size, 6, 9, 3)

        # Apply final color rules and balance
        balance = torch.sigmoid(self.algo_neural_balance).clamp(0.2, 0.8)
        light_result = self._blend_outputs(light_output, light_base, balance, is_dark=False)
        dark_result = self._blend_outputs(dark_output, dark_base, balance, is_dark=True)

        return torch.cat([
            light_result.view(batch_size, 162),
            dark_result.view(batch_size, 162)
        ], dim=1)

    def _blend_outputs(self, neural_output, algo_base, balance, is_dark=False):
        """Blend neural and algorithmic outputs"""
        batch_size = neural_output.shape[0]
        result = neural_output.clone()

        # Apply color contrast rules
        for i in range(6):
            if i % 3 == 1:  # Text colors
                bg_idx = (i // 3) * 3
                bg_lum = 0.299 * result[:, bg_idx, 0, 0] + 0.587 * result[:, bg_idx, 0, 1] + 0.114 * result[:, bg_idx, 0, 2]
                text_lum = 0.299 * result[:, i, 0, 0] + 0.587 * result[:, i, 0, 1] + 0.114 * result[:, i, 0, 2]

                min_contrast = 0.5 if is_dark else 0.4
                need_adjustment = torch.abs(bg_lum - text_lum) < min_contrast

                # Adjust text colors for contrast
                for b in range(batch_size):
                    if need_adjustment[b]:
                        if bg_lum[b] < 0.5:
                            result[b, i, 0, :] = torch.clamp(result[b, i, 0, :] * 0.7, 0, 1)
                        else:
                            result[b, i, 0, :] = torch.clamp(result[b, i, 0, :] + 0.3, 0, 1)

        return result

    def _create_light_base(self, x_reshaped):
        """Create light theme base colors"""
        batch_size = x_reshaped.shape[0]
        base_colors = torch.zeros(batch_size, 6, 9, 3, device=x_reshaped.device)

        for i in range(6):
            base_colors[:, i, 0, :] = x_reshaped[:, i, :]

        return base_colors

    def _create_dark_base(self, x_reshaped):
        """Create dark theme base colors with accent tint"""
        batch_size = x_reshaped.shape[0]
        base_colors = torch.zeros(batch_size, 6, 9, 3, device=x_reshaped.device)

        dark_bg = torch.clamp(self.dark_bg_level, 0.01, 0.4)
        light_text = torch.clamp(self.light_text_level, 0.6, 0.99)
        tint_strength = torch.clamp(self.accent_tint_strength, 0.0, 0.3)

        for i in range(6):
            if i % 3 == 0:  # Background colors
                accent_idx = (i // 3) * 3 + 2
                accent_color = x_reshaped[:, accent_idx, :]

                dark_base = dark_bg.unsqueeze(0).unsqueeze(-1) + 0.05 * (1.0 - x_reshaped[:, i, :])
                accent_influence = tint_strength.unsqueeze(0).unsqueeze(-1) * accent_color

                base_colors[:, i, 0, :] = torch.clamp(dark_base + accent_influence, 0, 1)

            elif i % 3 == 1:  # Text colors
                base_colors[:, i, 0, :] = light_text.unsqueeze(0).unsqueeze(-1) + 0.15 * (1.0 - x_reshaped[:, i, :])

            else:  # Accent colors
                base_colors[:, i, 0, :] = torch.clamp(x_reshaped[:, i, :] * 1.3, 0, 1)

        return base_colors