import torch

# Format output as CSS variables
def format_css_output(outputs):
    # List of color names in order
    color_names = [
        "primary-background", "primary-text", "primary-accent",
        "secondary-background", "secondary-text", "secondary-accent"
    ]

    # List of state suffixes
    state_suffixes = [
        "", "-hover", "-focus", "-disabled", "-border",
        "-gradient-a", "-gradient-b", "-gradient-c", "-gradient-d"
    ]

    # Split into light and dark themes
    light_theme = outputs[:162]
    dark_theme = outputs[162:]

    # Format light theme
    light_css = "/* Light-mode Theme */\n:root {\n"

    color_idx = 0
    for color_name in color_names:
        light_css += f"  /* {color_name.split('-')[0].capitalize()} */\n"
        for suffix in state_suffixes:
            var_name = f"--{color_name}{suffix}"
            r, g, b = light_theme[color_idx:color_idx+3]
            # Clamp values to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            light_css += f"  {var_name}: {r} {g} {b};\n"
            color_idx += 3
        light_css += "\n"

    light_css += "}\n\n"

    # Format dark theme
    dark_css = "/* Dark-mode Theme */\n.dark {\n"

    color_idx = 0
    for color_name in color_names:
        dark_css += f"  /* {color_name.split('-')[0].capitalize()} */\n"
        for suffix in state_suffixes:
            var_name = f"--{color_name}{suffix}"
            r, g, b = dark_theme[color_idx:color_idx+3]
            # Clamp values to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            dark_css += f"  {var_name}: {r} {g} {b};\n"
            color_idx += 3
        dark_css += "\n"

    dark_css += "}"

    return light_css + dark_css

# Function to generate CSS from inputs
def generate_color_palette(model, primary_bg, primary_text, primary_accent,
                          secondary_bg, secondary_text, secondary_accent):
    # Normalize inputs to [0-1]
    inputs = []
    for color in [primary_bg, primary_text, primary_accent,
                  secondary_bg, secondary_text, secondary_accent]:
        rgb = [int(x) for x in color.split()]
        inputs.extend([c/255.0 for c in rgb])

    # Prepare input for model
    device = next(model.parameters()).device
    input_tensor = torch.tensor([inputs], dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)

    # Convert back to 0-255 range
    outputs = (outputs * 255.0).cpu().numpy().astype(int)[0]

    # Format output as CSS variables
    css_output = format_css_output(outputs)
    return css_output
