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


class ColorPreservingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18

        # Encoder - extracts features from input colors
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        # Theme-specific branches with skip connections
        self.light_branch = nn.Sequential(
            nn.Linear(128 + self.input_dim, 192),  # Input colors concatenated
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(192, 162),
            nn.Sigmoid()
        )

        self.dark_branch = nn.Sequential(
            nn.Linear(128 + self.input_dim, 192),  # Input colors concatenated
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(192, 162),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract features
        features = self.encoder(x)

        # Light theme with skip connection
        light_input = torch.cat([features, x], dim=1)
        light_output_raw = self.light_branch(light_input)

        # Dark theme with skip connection
        dark_input = torch.cat([features, x], dim=1)
        dark_output_raw = self.dark_branch(dark_input)

        # Create new tensors instead of modifying in-place
        light_output = light_output_raw.clone().view(batch_size, 6, 9, 3)
        dark_output = dark_output_raw.clone().view(batch_size, 6, 9, 3)
        x_reshaped = x.view(batch_size, 6, 3)

        # Create final outputs using a functional approach rather than in-place operations
        light_result = self._apply_color_rules(light_output, x_reshaped, is_dark=False)
        dark_result = self._apply_color_rules(dark_output, x_reshaped, is_dark=True)

        # Combine outputs
        return torch.cat([light_result.view(batch_size, 162), dark_result.view(batch_size, 162)], dim=1)

    def _apply_color_rules(self, output, x_reshaped, is_dark=False):
        """Apply color rules without in-place operations"""
        batch_size = output.shape[0]
        result = torch.zeros_like(output)

        # Copy all values from output first
        result.copy_(output)

        # Apply base color rules
        for i in range(6):
            if not is_dark:
                # Light theme: base colors directly from input
                result[:, i, 0, :] = x_reshaped[:, i, :]
            else:
                if i % 3 == 0:  # Background colors
                  # Get corresponding accent color (using integer division and modulo)
                  accent_index = (i // 3) * 3 + 2  # Maps 0->2, 3->5 for accent colors
                  accent_color = x_reshaped[:, accent_index, :]

                  # Create dark background with accent tint
                  dark_base = 0.05 + 0.05 * (1.0 - x_reshaped[:, i, :])  # Dark inversion
                  accent_tint = 0.05 * accent_color  # Subtle accent influence

                  # Combine dark base with accent tint
                  result[:, i, 0, :] = torch.clamp(dark_base + accent_tint, 0, 1)
                elif i % 3 == 1:  # Text colors
                    result[:, i, 0, :] = 0.85 + 0.15 * (1.0 - x_reshaped[:, i, :])
                else:  # Accent colors
                    result[:, i, 0, :] = torch.clamp(x_reshaped[:, i, :] * 1.2, 0, 1)

        # Generate variations
        for i in range(6):
            base = result[:, i, 0, :].clone()

            if i % 3 == 0:  # Background colors
                if not is_dark:
                    # Light theme variations
                    result[:, i, 1, :] = torch.clamp(base - 0.04, 0, 1)  # hover
                    result[:, i, 2, :] = torch.clamp(base - 0.08, 0, 1)  # focus
                    result[:, i, 3, :] = torch.clamp(base + 0.02, 0, 1)  # disabled
                    result[:, i, 4, :] = torch.clamp(base - 0.15, 0, 1)  # border
                else:
                    # Dark theme variations
                    result[:, i, 1, :] = torch.clamp(base + 0.04, 0, 1)  # hover
                    result[:, i, 2, :] = torch.clamp(base + 0.08, 0, 1)  # focus
                    result[:, i, 3, :] = torch.clamp(base - 0.04, 0, 1)  # disabled
                    result[:, i, 4, :] = torch.clamp(base - 0.02, 0, 1)  # border

            elif i % 3 == 1:  # Text colors
                if not is_dark:
                    result[:, i, 1, :] = torch.clamp(base - 0.04, 0, 1)  # hover
                    result[:, i, 2, :] = torch.clamp(base - 0.08, 0, 1)  # focus
                    result[:, i, 3, :] = torch.clamp(base + 0.17, 0, 1)  # disabled
                    result[:, i, 4, :] = torch.clamp(base - 0.02, 0, 1)  # border
                else:
                    result[:, i, 1, :] = torch.clamp(base + 0.04, 0, 1)  # hover
                    result[:, i, 2, :] = torch.clamp(base + 0.08, 0, 1)  # focus
                    result[:, i, 3, :] = torch.clamp(base - 0.15, 0, 1)  # disabled
                    result[:, i, 4, :] = torch.clamp(base - 0.05, 0, 1)  # border

            else:  # Accent colors
                if not is_dark:
                    result[:, i, 1, :] = torch.clamp(base - 0.05, 0, 1)  # hover
                    result[:, i, 2, :] = torch.clamp(base - 0.1, 0, 1)   # focus
                    result[:, i, 3, :] = torch.clamp(base + 0.12, 0, 1)  # disabled
                    result[:, i, 4, :] = torch.clamp(base - 0.12, 0, 1)  # border
                else:
                    result[:, i, 1, :] = torch.clamp(base + 0.07, 0, 1)  # hover
                    result[:, i, 2, :] = torch.clamp(base + 0.14, 0, 1)  # focus
                    result[:, i, 3, :] = torch.clamp(base - 0.12, 0, 1)  # disabled
                    result[:, i, 4, :] = torch.clamp(base - 0.08, 0, 1)  # border

            # Gradients
            result[:, i, 5, :] = base  # gradient-a = base
            if not is_dark:
                result[:, i, 6, :] = torch.clamp(base - 0.05, 0, 1)  # gradient-b
                result[:, i, 7, :] = torch.clamp(base - 0.1, 0, 1)   # gradient-c
                result[:, i, 8, :] = torch.clamp(base - 0.15, 0, 1)  # gradient-d
            else:
                result[:, i, 6, :] = torch.clamp(base + 0.05, 0, 1)  # gradient-b
                result[:, i, 7, :] = torch.clamp(base + 0.1, 0, 1)   # gradient-c
                result[:, i, 8, :] = torch.clamp(base + 0.15, 0, 1)  # gradient-d

        return result
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class EnhancedColorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18

        # Deeper encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        # Color space transformer (learn color relationships instead of hard-coding them)
        self.color_transform = nn.Sequential(
            nn.Linear(256 + self.input_dim, 192),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )

        # Theme-specific branches with residual connections
        self.light_branch = nn.ModuleList([
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Linear(128, 162),
            nn.Sigmoid()
        ])

        self.dark_branch = nn.ModuleList([
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Linear(128, 162),
            nn.Sigmoid()
        ])

        # Learnable color relationship parameters (instead of fixed rules)
        self.light_variations = nn.Parameter(torch.randn(6, 5, 3) * 0.02)
        self.dark_variations = nn.Parameter(torch.randn(6, 5, 3) * 0.02)

        # Base color influence parameters (how much of original color to keep)
        self.light_base_influence = nn.Parameter(torch.ones(6) * 0.8)  # Start with 80% influence
        self.dark_base_influence = nn.Parameter(torch.ones(6) * 0.3)   # Start with 30% influence

        # Color blending factors (how much to blend between adjacent colors)
        self.light_blend = nn.Parameter(torch.zeros(6, 6) + 0.05)  # Small initial cross-influence
        self.dark_blend = nn.Parameter(torch.zeros(6, 6) + 0.1)    # Slightly larger cross-influence for dark

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract features
        features = self.encoder(x)

        # Common color transform with skip connection
        color_input = torch.cat([features, x], dim=1)
        color_features = self.color_transform(color_input)

        # Light theme generation
        light_output = color_features
        for layer in self.light_branch[:-2]:
            light_output = layer(light_output)
        light_output = self.light_branch[-2](light_output)
        light_output = self.light_branch[-1](light_output).view(batch_size, 6, 9, 3)

        # Dark theme generation
        dark_output = color_features
        for layer in self.dark_branch[:-2]:
            dark_output = layer(dark_output)
        dark_output = self.dark_branch[-2](dark_output)
        dark_output = self.dark_branch[-1](dark_output).view(batch_size, 6, 9, 3)

        # Apply learned variations
        light_result = self._apply_learned_variations(
            light_output,
            x.view(batch_size, 6, 3),
            self.light_variations,
            self.light_base_influence,
            self.light_blend,
            is_dark=False
        )

        dark_result = self._apply_learned_variations(
            dark_output,
            x.view(batch_size, 6, 3),
            self.dark_variations,
            self.dark_base_influence,
            self.dark_blend,
            is_dark=True
        )

        return torch.cat([light_result.view(batch_size, 162), dark_result.view(batch_size, 162)], dim=1)

    def _apply_learned_variations(self, output, x_reshaped, variations, base_influence, blend_factors, is_dark=False):
        """Apply learned color variations with blending between colors"""
        batch_size = output.shape[0]
        result = torch.zeros_like(output)

        # First, handle the base colors (index 0 of each color group)
        for i in range(6):
            # Get base color influence (how much of original color to keep)
            influence = torch.sigmoid(base_influence[i]).unsqueeze(0).unsqueeze(-1)

            # For dark theme, we start with a darker base
            if is_dark and i % 3 == 0:  # Background colors
                base = 0.15 + influence * x_reshaped[:, i, :]  # Darker starting point
            elif is_dark and i % 3 == 1:  # Text colors
                base = 0.85 + influence * (1.0 - x_reshaped[:, i, :])  # Lighter for text
            elif is_dark and i % 3 == 2:  # Accent colors
                base = torch.clamp(x_reshaped[:, i, :] * (1.0 + influence), 0, 1)  # More vivid
            else:  # Light theme - more directly influenced by original
                base = x_reshaped[:, i, :]

            # Apply cross-color blending
            for j in range(6):
                if i != j:  # Don't blend with self
                    # Get blend factor for this color pair
                    blend = torch.sigmoid(blend_factors[i, j]).unsqueeze(0).unsqueeze(-1)
                    # Add influence from other color
                    base = base * (1 - blend) + x_reshaped[:, j, :] * blend

            # Set the base color
            result[:, i, 0, :] = base

            # Now generate variations using learned offsets
            for v in range(4):  # 4 variations (hover, focus, disabled, border)
                # Get the variation offset and apply it
                offset = torch.tanh(variations[i, v, :]).unsqueeze(0)  # Range -1 to 1

                if is_dark:
                    # For dark theme, positive offsets make colors lighter
                    variation = torch.clamp(base + offset * 0.2, 0, 1)
                else:
                    # For light theme, positive offsets make colors darker
                    variation = torch.clamp(base - offset * 0.2, 0, 1)

                result[:, i, v+1, :] = variation

            # Generate gradient variations
            # First gradient color is the base
            result[:, i, 5, :] = base

            # Generate 3 additional gradient colors
            for g in range(3):
                # Get gradient from base with different intensity
                if is_dark:
                    grad_value = torch.clamp(base + (g+1) * 0.05, 0, 1)
                else:
                    grad_value = torch.clamp(base - (g+1) * 0.05, 0, 1)

                result[:, i, g+6, :] = grad_value

        return result
