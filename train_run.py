import torch
import torch.optim as optim
from model.utils import generate_color_palette
from model.hybrid import ImprovedHybridColorGenerator
from model.trainer import color_harmony_loss, train_model


data_path = "dataset/extracted_css_vars.json"

model = ImprovedHybridColorGenerator()
# model = AutoEncoderColorGenerator()

lr=0.0001
epochs = 500

loss_func = color_harmony_loss

# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#   optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
# )

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
  optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Alternative: One-cycle learning rate
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#   optimizer, max_lr=0.01, total_steps=epochs * len(train_loader)
# )

model = train_model(
    model=model,
    batch_size=16,
    data_path=data_path,
    loss_func=loss_func,
    scheduler=scheduler,
    optimizer=optimizer,
    epochs=epochs, lr=lr,
)


# Example usage
primary_bg = "255 255 255"
primary_text = "17 24 39"
primary_accent = "109 40 217"
secondary_bg = "243 244 246"
secondary_text = "107 114 128"
secondary_accent = "239 68 68"

css = generate_color_palette(model, primary_bg, primary_text, primary_accent,
                               secondary_bg, secondary_text, secondary_accent)
print(css)