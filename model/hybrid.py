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


class HybridColorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18

        # Theme detection network
        self.theme_detector = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output: probability of being light theme
        )

        # Feature extraction network
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

        # Refinement networks for each theme
        self.light_refiner = nn.Sequential(
            nn.Linear(256 + self.input_dim + 54, 256),  # Add inverted base colors
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 162),
            nn.Sigmoid()
        )

        self.dark_refiner = nn.Sequential(
            nn.Linear(256 + self.input_dim + 54, 256),  # Add inverted base colors
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 162),
            nn.Sigmoid()
        )

        # Learnable adjustment parameters
        self.accent_tint_strength = nn.Parameter(torch.ones(1) * 0.05)  # 5% initial tint
        self.dark_bg_level = nn.Parameter(torch.ones(1) * 0.12)         # Initial dark level
        self.light_text_level = nn.Parameter(torch.ones(1) * 0.85)      # Initial light level

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, 6, 3)

        # 1. Detect theme (though we'll generate both regardless)
        theme_prob = self.theme_detector(x)

        # 2. Extract features
        features = self.encoder(x)

        # 3. Create initial inversions using algorithmic rules
        light_base = self._create_light_base(x_reshaped)
        dark_base = self._create_dark_base(x_reshaped)

        # 4. Concatenate features, original input, and base colors
        light_input = torch.cat([
            features,
            x,
            light_base.view(batch_size, -1)
        ], dim=1)

        dark_input = torch.cat([
            features,
            x,
            dark_base.view(batch_size, -1)
        ], dim=1)

        # 5. Generate refined palettes
        light_output = self.light_refiner(light_input).view(batch_size, 6, 9, 3)
        dark_output = self.dark_refiner(dark_input).view(batch_size, 6, 9, 3)

        # 6. Apply color rules for stability and guidance
        light_result = self._apply_color_rules(light_output, x_reshaped, light_base, is_dark=False)
        dark_result = self._apply_color_rules(dark_output, x_reshaped, dark_base, is_dark=True)

        # 7. Return combined result
        return torch.cat([
            light_result.view(batch_size, 162),
            dark_result.view(batch_size, 162)
        ], dim=1)

    def _create_light_base(self, x_reshaped):
        """Create light theme base colors (mostly direct use of input)"""
        batch_size = x_reshaped.shape[0]
        base_colors = torch.zeros(batch_size, 6, 3, 3, device=x_reshaped.device)

        # Set base colors (directly from input)
        for i in range(6):
            base_colors[:, i, 0, :] = x_reshaped[:, i, :]

        return base_colors

    def _create_dark_base(self, x_reshaped):
        """Create dark theme base colors with accent tinting"""
        batch_size = x_reshaped.shape[0]
        base_colors = torch.zeros(batch_size, 6, 3, 3, device=x_reshaped.device)

        # Get adjustable parameters
        dark_bg = torch.clamp(self.dark_bg_level, 0.05, 0.25)
        light_text = torch.clamp(self.light_text_level, 0.75, 0.95)
        tint_strength = torch.clamp(self.accent_tint_strength, 0.01, 0.2)

        for i in range(6):
            if i % 3 == 0:  # Background colors with accent tint
                # Find corresponding accent color
                accent_idx = (i // 3) * 3 + 2
                accent_color = x_reshaped[:, accent_idx, :]

                # Create dark bg with accent tint
                dark_base = dark_bg.unsqueeze(0).unsqueeze(-1) + 0.05 * (1.0 - x_reshaped[:, i, :])
                accent_influence = tint_strength.unsqueeze(0).unsqueeze(-1) * accent_color

                base_colors[:, i, 0, :] = torch.clamp(dark_base + accent_influence, 0, 1)

            elif i % 3 == 1:  # Text colors (invert and make light)
                base_colors[:, i, 0, :] = light_text.unsqueeze(0).unsqueeze(-1) + 0.15 * (1.0 - x_reshaped[:, i, :])

            else:  # Accent colors (make more vibrant)
                base_colors[:, i, 0, :] = torch.clamp(x_reshaped[:, i, :] * 1.3, 0, 1)

        return base_colors

    def _apply_color_rules(self, output, x_reshaped, base_colors, is_dark=False):
        """Apply color rules to ensure harmony while allowing model flexibility"""
        batch_size = output.shape[0]
        result = output.clone()  # Start with model output

        # Ensure base colors are preserved with some influence from the model
        for i in range(6):
            # Mix 70% of algorithmic base with 30% of model prediction for base colors
            result[:, i, 0, :] = 0.7 * base_colors[:, i, 0, :] + 0.3 * output[:, i, 0, :]

            # Ensure proper contrast between text and background
            if i % 3 == 1:  # Text colors
                bg_idx = (i // 3) * 3  # Corresponding background

                # Calculate luminance
                bg_lum = 0.299 * result[:, bg_idx, 0, 0] + 0.587 * result[:, bg_idx, 0, 1] + 0.114 * result[:, bg_idx, 0, 2]
                text_lum = 0.299 * result[:, i, 0, 0] + 0.587 * result[:, i, 0, 1] + 0.114 * result[:, i, 0, 2]

                # Ensure minimum contrast ratio based on theme
                min_contrast = 0.6 if is_dark else 0.5
                lum_diff = torch.abs(bg_lum - text_lum)

                # Adjust text color where contrast is insufficient
                need_adjustment = lum_diff < min_contrast
                if torch.any(need_adjustment):
                    # Push text colors to be lighter or darker based on background
                    for b in range(batch_size):
                        if need_adjustment[b]:
                            if (bg_lum[b] < 0.5 and not is_dark) or (bg_lum[b] >= 0.5 and is_dark):
                                # Make text darker
                                result[b, i, 0, :] = torch.clamp(result[b, i, 0, :] * 0.7, 0, 1)
                            else:
                                # Make text lighter
                                result[b, i, 0, :] = torch.clamp(result[b, i, 0, :] + 0.3, 0, 1)

        return result
    

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.block(x) + x)

class ImprovedHybridColorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 18

        # Theme detection network - now actually used to determine input theme
        self.theme_detector = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output > 0.5 means light theme, < 0.5 means dark theme
        )

        # Feature extraction with residual blocks
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128, 0.15),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256, 0.15),
            nn.Linear(256, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
        )

        # Enhanced refinement networks with residual blocks
        refiner_input_size = 384 + self.input_dim + 54

        self.light_refiner = nn.Sequential(
            nn.Linear(refiner_input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            ResidualBlock(384, 0.1),
            nn.Linear(384, 162),
            nn.Sigmoid()
        )

        self.dark_refiner = nn.Sequential(
            nn.Linear(refiner_input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            ResidualBlock(384, 0.1),
            nn.Linear(384, 162),
            nn.Sigmoid()
        )

        # Learnable parameters with wider ranges
        self.accent_tint_strength = nn.Parameter(torch.ones(1) * 0.05)  # Initial 5% tint
        self.dark_bg_level = nn.Parameter(torch.ones(1) * 0.12)         # Initial dark level
        self.light_text_level = nn.Parameter(torch.ones(1) * 0.85)      # Initial light level

        # Learnable algorithmic influence (per color type)
        self.algorithmic_weight = nn.Parameter(torch.ones(6) * 0.5)     # Start at 50% influence

        # Learnable contrast requirements
        self.min_contrast_dark = nn.Parameter(torch.ones(1) * 0.55)     # Initial dark contrast
        self.min_contrast_light = nn.Parameter(torch.ones(1) * 0.45)    # Initial light contrast

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, 6, 3)

        # 1. Detect theme (now actually used)
        theme_probs = self.theme_detector(x)
        is_light_theme = theme_probs > 0.5  # Batch of booleans

        # 2. Extract features
        features = self.encoder(x)

        # 3. Create theme bases based on detection
        # Initialize with zeros tensors for both themes
        light_base = torch.zeros(batch_size, 6, 3, 3, device=x.device)
        dark_base = torch.zeros(batch_size, 6, 3, 3, device=x.device)

        # Process each sample based on its detected theme
        for b in range(batch_size):
            if is_light_theme[b]:
                # Input is light - use it directly for light theme
                for i in range(6):
                    light_base[b, i, 0, :] = x_reshaped[b, i, :]

                # Create dark theme algorithmically
                dark_base[b] = self._create_dark_base_single(x_reshaped[b])
            else:
                # Input is dark - use it directly for dark theme
                for i in range(6):
                    dark_base[b, i, 0, :] = x_reshaped[b, i, :]

                # Create light theme algorithmically
                light_base[b] = self._create_light_base_single(x_reshaped[b])

        # 4. Concatenate features, original input, and base colors
        light_input = torch.cat([
            features,
            x,
            light_base.view(batch_size, -1)
        ], dim=1)

        dark_input = torch.cat([
            features,
            x,
            dark_base.view(batch_size, -1)
        ], dim=1)

        # 5. Generate refined palettes
        light_output = self.light_refiner(light_input).view(batch_size, 6, 9, 3)
        dark_output = self.dark_refiner(dark_input).view(batch_size, 6, 9, 3)

        # 6. Apply color rules with adaptive weights based on detected theme
        light_result = self._apply_adaptive_rules(light_output, x_reshaped, light_base, is_light_theme, is_dark=False)
        dark_result = self._apply_adaptive_rules(dark_output, x_reshaped, dark_base, is_light_theme, is_dark=True)

        # 7. Return combined result
        return torch.cat([
            light_result.view(batch_size, 162),
            dark_result.view(batch_size, 162)
        ], dim=1)

    def _create_light_base_single(self, single_input):
        """Create light theme base colors from a dark input"""
        base_colors = torch.zeros(6, 3, 3, device=single_input.device)

        for i in range(6):
            if i % 3 == 0:  # Background colors (make light)
                base_colors[i, 0, :] = 0.9 - 0.2 * (1.0 - single_input[i, :])
            elif i % 3 == 1:  # Text colors (make dark)
                base_colors[i, 0, :] = 0.1 + 0.2 * single_input[i, :]
            else:  # Accent colors (maintain but adjust)
                base_colors[i, 0, :] = torch.clamp(single_input[i, :] * 0.9, 0, 1)

        return base_colors

    def _create_dark_base_single(self, single_input):
        """Create dark theme base colors from a light input"""
        base_colors = torch.zeros(6, 3, 3, device=single_input.device)

        # Get adjustable parameters with wider ranges
        dark_bg = torch.clamp(self.dark_bg_level, 0.01, 0.4)
        light_text = torch.clamp(self.light_text_level, 0.6, 0.99)
        tint_strength = torch.clamp(self.accent_tint_strength, 0.0, 0.3)

        for i in range(6):
            if i % 3 == 0:  # Background colors with accent tint
                # Find corresponding accent color
                accent_idx = (i // 3) * 3 + 2
                accent_color = single_input[accent_idx, :]

                # Create dark bg with accent tint
                dark_base = dark_bg.unsqueeze(0) + 0.05 * (1.0 - single_input[i, :])
                accent_influence = tint_strength.unsqueeze(0) * accent_color

                base_colors[i, 0, :] = torch.clamp(dark_base + accent_influence, 0, 1)

            elif i % 3 == 1:  # Text colors (invert and make light)
                base_colors[i, 0, :] = light_text.unsqueeze(0) + 0.15 * (1.0 - single_input[i, :])

            else:  # Accent colors (make more vibrant)
                base_colors[i, 0, :] = torch.clamp(single_input[i, :] * 1.3, 0, 1)

        return base_colors

    def _apply_adaptive_rules(self, output, x_reshaped, base_colors, is_light_theme, is_dark=False):
        """Apply color rules with adaptive weights based on input theme"""
        batch_size = output.shape[0]
        result = output.clone()  # Start with model output

        # Get learnable contrast requirements
        min_contrast = torch.clamp(
            self.min_contrast_dark if is_dark else self.min_contrast_light,
            0.3, 0.7
        )

        # Ensure base colors are preserved with learnable influence
        for i in range(6):
            # Get learnable algorithmic weight for this color type
            alg_weight = torch.clamp(torch.sigmoid(self.algorithmic_weight[i]), 0.2, 0.8)

            for b in range(batch_size):
                # Use more algorithmic influence for the theme we're generating
                # and less for the theme we're preserving
                if (is_light_theme[b] and is_dark) or (not is_light_theme[b] and not is_dark):
                    # This is the theme we're generating - use more algorithmic influence
                    weight = alg_weight
                else:
                    # This is the theme we're preserving - use less algorithmic influence
                    weight = alg_weight * 0.5  # Halve the algorithmic influence

                # Mix with learnable weights (allowing more model influence)
                result[b, i, 0, :] = weight * base_colors[b, i, 0, :] + (1-weight) * output[b, i, 0, :]

            # Ensure proper contrast between text and background
            if i % 3 == 1:  # Text colors
                bg_idx = (i // 3) * 3  # Corresponding background

                # Calculate luminance
                bg_lum = 0.299 * result[:, bg_idx, 0, 0] + 0.587 * result[:, bg_idx, 0, 1] + 0.114 * result[:, bg_idx, 0, 2]
                text_lum = 0.299 * result[:, i, 0, 0] + 0.587 * result[:, i, 0, 1] + 0.114 * result[:, i, 0, 2]

                # Use learnable minimum contrast
                lum_diff = torch.abs(bg_lum - text_lum)

                # Adjust text color where contrast is insufficient
                need_adjustment = lum_diff < min_contrast
                if torch.any(need_adjustment):
                    # Push text colors to be lighter or darker based on background
                    for b in range(batch_size):
                        if need_adjustment[b]:
                            if (bg_lum[b] < 0.5 and not is_dark) or (bg_lum[b] >= 0.5 and is_dark):
                                # Make text darker
                                result[b, i, 0, :] = torch.clamp(result[b, i, 0, :] * 0.7, 0, 1)
                            else:
                                # Make text lighter
                                result[b, i, 0, :] = torch.clamp(result[b, i, 0, :] + 0.3, 0, 1)

        return result
    