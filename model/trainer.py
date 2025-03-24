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


# Custom Dataset class
class ColorPaletteDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract input colors
        input_colors = []
        for key in ["primary-background", "primary-text", "primary-accent",
                    "secondary-background", "secondary-text", "secondary-accent"]:
            rgb = [int(x) for x in item["input"][key].split()]
            input_colors.extend(rgb)

        # Extract output colors
        output_colors = []
        # First light theme
        for prefix in ["primary-background", "primary-text", "primary-accent",
                      "secondary-background", "secondary-text", "secondary-accent"]:
            for suffix in ["", "-hover", "-focus", "-disabled", "-border",
                          "-gradient-a", "-gradient-b", "-gradient-c", "-gradient-d"]:
                key = f"{prefix}{suffix}"
                if key in item["output"]["root"]:
                    rgb = [int(x) for x in item["output"]["root"][key].split()]
                    output_colors.extend(rgb)
                else:
                    # Handle missing keys (fallback to base color)
                    base_key = prefix
                    rgb = [int(x) for x in item["output"]["root"][base_key].split()]
                    output_colors.extend(rgb)

        # Then dark theme
        for prefix in ["primary-background", "primary-text", "primary-accent",
                      "secondary-background", "secondary-text", "secondary-accent"]:
            for suffix in ["", "-hover", "-focus", "-disabled", "-border",
                          "-gradient-a", "-gradient-b", "-gradient-c", "-gradient-d"]:
                key = f"{prefix}{suffix}"
                if key in item["output"]["dark"]:
                    rgb = [int(x) for x in item["output"]["dark"][key].split()]
                    output_colors.extend(rgb)
                else:
                    # Handle missing keys (fallback to base color)
                    base_key = prefix
                    rgb = [int(x) for x in item["output"]["dark"][base_key].split()]
                    output_colors.extend(rgb)

        return {
            'input': torch.tensor(input_colors, dtype=torch.float32) / 255.0,  # Normalize to [0, 1]
            'output': torch.tensor(output_colors, dtype=torch.float32) / 255.0
        }


# Calculate contrast ratio between text and background
def calculate_contrast_ratio(colors):
    # Extract text and background
    # This is a simplified version - you'd adapt this to your specific layout
    # Assuming colors is shaped as [batch_size, num_colors * 3]
    batch_size = colors.shape[0]
    loss = 0.0

    for i in range(batch_size):
        # Simplified example - actual logic would depend on your color layout
        bg_colors = colors[i, 0:3].unsqueeze(0)  # First color as background
        text_colors = colors[i, 3:6].unsqueeze(0)  # Second color as text

        # Calculate luminance (simplified)
        bg_lum = 0.2126 * bg_colors[:, 0] + 0.7152 * bg_colors[:, 1] + 0.0722 * bg_colors[:, 2]
        text_lum = 0.2126 * text_colors[:, 0] + 0.7152 * text_colors[:, 1] + 0.0722 * text_colors[:, 2]

        # Calculate contrast ratio
        lighter = torch.max(bg_lum, text_lum)
        darker = torch.min(bg_lum, text_lum)
        contrast = (lighter + 0.05) / (darker + 0.05)

        # Penalize contrast below 4.5 (WCAG AA standard)
        loss += torch.relu(4.5 - contrast)

    return loss / batch_size

# Calculate consistency between different states of the same color
def calculate_state_consistency(colors):
    # This is a simplified function - adapt to your color structure
    # Assuming proper ordering of colors in the tensor
    batch_size = colors.shape[0]
    colors_reshaped = colors.view(batch_size, -1, 3)
    loss = 0.0

    # Check that hover is slightly darker than base, focus darker than hover, etc.
    for i in range(batch_size):
        for color_idx in range(0, colors_reshaped.shape[1], 9):  # 9 states per color
            base = colors_reshaped[i, color_idx]
            hover = colors_reshaped[i, color_idx + 1]
            focus = colors_reshaped[i, color_idx + 2]

            # Penalize if hover isn't slightly darker/more saturated than base
            # Simple luminance check (actual implementation would be more sophisticated)
            base_lum = torch.mean(base)
            hover_lum = torch.mean(hover)
            focus_lum = torch.mean(focus)

            # Penalize incorrect relationships
            loss += torch.relu(hover_lum - base_lum)  # hover should be darker
            loss += torch.relu(hover_lum - focus_lum)  # focus should be darker than hover

    return loss / batch_size


# Improved Loss Function with Color Preservation
def color_harmony_loss(pred, target, inputs):
    # Split outputs
    batch_size = pred.shape[0]
    light_pred = pred[:, :162].view(batch_size, 6, 9, 3)
    dark_pred = pred[:, 162:].view(batch_size, 6, 9, 3)
    target_light = target[:, :162].view(batch_size, 6, 9, 3)
    target_dark = target[:, 162:].view(batch_size, 6, 9, 3)
    inputs_reshaped = inputs.view(batch_size, 6, 3)

    # 1. Base MSE loss
    mse_loss = F.mse_loss(pred, target)

    # 2. Color preservation loss - weighted heavily
    color_preservation_loss = 0
    for i in range(6):
        # Light theme base colors should match inputs
        color_preservation_loss += F.mse_loss(light_pred[:, i, 0, :], inputs_reshaped[:, i, :]) * 5

    # 3. Relationship loss - between variations of same color
    relationship_loss = 0
    for i in range(6):
        for theme_pred in [light_pred, dark_pred]:
            base = theme_pred[:, i, 0, :]

            # Each variation should be related to its base color
            for j in range(1, 9):
                variation = theme_pred[:, i, j, :]
                if i % 3 == 0:  # Background colors
                    # Background hover/focus should be related but different
                    if j in [1, 2]:  # hover, focus
                        if theme_pred is light_pred:
                            # Should be darker in light theme
                            relationship_loss += torch.mean(torch.relu(variation - base))
                        else:
                            # Should be lighter in dark theme
                            relationship_loss += torch.mean(torch.relu(base - variation))

            # Gradient should flow smoothly
            for j in range(5, 8):
                curr = theme_pred[:, i, j, :]
                next_var = theme_pred[:, i, j+1, :]
                if theme_pred is light_pred:
                    # Should get progressively darker
                    relationship_loss += torch.mean(torch.relu(next_var - curr))
                else:
                    # Should get progressively lighter
                    relationship_loss += torch.mean(torch.relu(curr - next_var))

    # 4. Contrast loss - ensure text and backgrounds have good contrast
    contrast_loss = 0
    for theme_pred in [light_pred, dark_pred]:
        # Primary background to primary text contrast
        bg = theme_pred[:, 0, 0, :]  # Primary background
        text = theme_pred[:, 1, 0, :]  # Primary text
        luminance_bg = 0.299 * bg[:, 0] + 0.587 * bg[:, 1] + 0.114 * bg[:, 2]
        luminance_text = 0.299 * text[:, 0] + 0.587 * text[:, 1] + 0.114 * text[:, 2]
        contrast = torch.abs(luminance_bg - luminance_text)
        # contrast_loss += torch.mean(torch.relu(0.5 - contrast))
        # contrast_loss += torch.mean(torch.relu(contrast - 0.9))

        # Secondary background to secondary text contrast
        sec_bg = theme_pred[:, 3, 0, :]  # Secondary background
        sec_text = theme_pred[:, 4, 0, :]  # Secondary text
        luminance_sec_bg = 0.299 * sec_bg[:, 0] + 0.587 * sec_bg[:, 1] + 0.114 * sec_bg[:, 2]
        luminance_sec_text = 0.299 * sec_text[:, 0] + 0.587 * sec_text[:, 1] + 0.114 * sec_text[:, 2]
        sec_contrast = torch.abs(luminance_sec_bg - luminance_sec_text)
        # contrast_loss += torch.mean(torch.relu(0.5 - sec_contrast))
        # contrast_loss += torch.mean(torch.relu(sec_contrast - 0.9))

    # 4. Contrast loss - ensure text and backgrounds have good contrast
    # contrast_loss = 0
    # for theme_pred in [light_pred, dark_pred]:
    #     # Primary background to primary text contrast
    #     bg = theme_pred[:, 0, 0, :]  # Primary background
    #     text = theme_pred[:, 1, 0, :]  # Primary text
    #     luminance_bg = 0.299 * bg[:, 0] + 0.587 * bg[:, 1] + 0.114 * bg[:, 2]
    #     luminance_text = 0.299 * text[:, 0] + 0.587 * text[:, 1] + 0.114 * text[:, 2]
    #     contrast = torch.abs(luminance_bg - luminance_text)

    #     # Penalize if contrast is too low (below 0.5)
    #     min_contrast_loss = torch.mean(torch.relu(0.5 - contrast))

    #     # Penalize if contrast is too high (above 0.7)
    #     max_contrast_loss = torch.mean(torch.relu(contrast - 0.7))

    #     contrast_loss += min_contrast_loss + max_contrast_loss

    #     # Secondary background to secondary text contrast
    #     sec_bg = theme_pred[:, 3, 0, :]  # Secondary background
    #     sec_text = theme_pred[:, 4, 0, :]  # Secondary text
    #     luminance_sec_bg = 0.299 * sec_bg[:, 0] + 0.587 * sec_bg[:, 1] + 0.114 * sec_bg[:, 2]
    #     luminance_sec_text = 0.299 * sec_text[:, 0] + 0.587 * sec_text[:, 1] + 0.114 * sec_text[:, 2]
    #     sec_contrast = torch.abs(luminance_sec_bg - luminance_sec_text)

    #     # Penalize if secondary contrast is too low or too high
    #     sec_min_contrast_loss = torch.mean(torch.relu(0.5 - sec_contrast))
    #     sec_max_contrast_loss = torch.mean(torch.relu(sec_contrast - 0.7))

    #     contrast_loss += sec_min_contrast_loss + sec_max_contrast_loss

    # 5. Theme difference loss - ensure dark theme is actually dark and light theme is light
    theme_difference_loss = 0
    for i in range(0, 6, 3):  # Background indices: 0, 3
        light_bg = light_pred[:, i, 0, :]
        dark_bg = dark_pred[:, i, 0, :]

        # Light background should be lighter than dark background
        light_luminance = 0.299 * light_bg[:, 0] + 0.587 * light_bg[:, 1] + 0.114 * light_bg[:, 2]
        dark_luminance = 0.299 * dark_bg[:, 0] + 0.587 * dark_bg[:, 1] + 0.114 * dark_bg[:, 2]

        # Penalize if light bg isn't lighter than dark bg
        theme_difference_loss += torch.mean(torch.relu(dark_luminance - light_luminance + 0.4))

    # Weighted combination
    return (mse_loss * 0.5 +
            color_preservation_loss * 2.0 +
            relationship_loss * 0.3 +
            contrast_loss * 1.0 +
            theme_difference_loss * 1.0)


# Training function
def train_model(model, data_path, scheduler, optimizer, loss_func=color_harmony_loss, batch_size=32, epochs=100, lr=0.001):
    # Load and split dataset
    dataset = ColorPaletteDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    # Training and validation history
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets, inputs)
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)

                outputs = model(inputs)
                loss = color_harmony_loss(outputs, targets, inputs)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_color_palette_model.pth')
            patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

    # Load best model
    model.load_state_dict(torch.load('best_color_palette_model.pth'))

    return model

