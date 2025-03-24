import json
import random

def generate_base_palette(input_tailwind_colorset: str="dataset/tailwind-colors-python-format.json", sample_size: int=25, output_file: str="dataset/color_palette_dataset.json" ):

    # Load the Tailwind colors from the JSON file
    with open(input_tailwind_colorset, 'r') as f:
        TAILWIND_COLORS_RAW = json.load(f)

    # Validate and clean the Tailwind colors
    TAILWIND_COLORS = {}
    for color_name, shades in TAILWIND_COLORS_RAW.items():
        TAILWIND_COLORS[color_name] = {}
        for shade, rgb_str in shades.items():
            # Parse RGB values
            try:
                rgb_values = [int(x.strip()) for x in rgb_str.split()]
                # Ensure valid RGB values (0-255)
                valid_rgb = [max(0, min(255, val)) for val in rgb_values]
                TAILWIND_COLORS[color_name][shade] = f"{valid_rgb[0]} {valid_rgb[1]} {valid_rgb[2]}"
            except (ValueError, IndexError):
                # Skip invalid colors
                print(f"Skipping invalid color: {color_name}-{shade}: {rgb_str}")

    # Define standard Tailwind shades
    SHADES = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900', '950']

    # Define light, medium and dark shade groups
    LIGHT_SHADES = ['50', '100', '200']
    MEDIUM_SHADES = ['400', '500', '600']
    DARK_SHADES = ['800', '900', '950']

    # Define color groups
    NEUTRALS = ['slate', 'gray', 'zinc', 'neutral', 'stone']
    COLORS = ['red', 'orange', 'amber', 'yellow', 'lime', 'green', 'emerald', 
            'teal', 'cyan', 'sky', 'blue', 'indigo', 'violet', 'purple', 
            'fuchsia', 'pink', 'rose']

    # Verify all colors exist
    COLORS = [c for c in COLORS if c in TAILWIND_COLORS]
    NEUTRALS = [n for n in NEUTRALS if n in TAILWIND_COLORS]

    # Define color wheel for strategies that rely on color relationships
    COLOR_WHEEL = [c for c in ['red', 'orange', 'amber', 'yellow', 'lime', 'green', 'emerald', 
                'teal', 'cyan', 'sky', 'blue', 'indigo', 'violet', 'purple', 
                'fuchsia', 'pink', 'rose'] if c in TAILWIND_COLORS]

    def color_exists(color, shade):
        """Check if a color and shade exists in our color dictionary"""
        return color in TAILWIND_COLORS and shade in TAILWIND_COLORS[color]

    def get_analogous_colors(color):
        """Get colors adjacent to the given color on the color wheel"""
        if color not in COLOR_WHEEL:
            return [COLOR_WHEEL[0], COLOR_WHEEL[1]]
        
        idx = COLOR_WHEEL.index(color)
        prev_idx = (idx - 1) % len(COLOR_WHEEL)
        next_idx = (idx + 1) % len(COLOR_WHEEL)
        return [COLOR_WHEEL[prev_idx], COLOR_WHEEL[next_idx]]

    def get_complementary_color(color):
        """Get the color opposite on the color wheel"""
        if color not in COLOR_WHEEL:
            return COLOR_WHEEL[0]
            
        idx = COLOR_WHEEL.index(color)
        opposite_idx = (idx + len(COLOR_WHEEL) // 2) % len(COLOR_WHEEL)
        return COLOR_WHEEL[opposite_idx]

    def get_triadic_colors(color):
        """Get three colors equally spaced on the color wheel"""
        if color not in COLOR_WHEEL:
            return [COLOR_WHEEL[0], COLOR_WHEEL[len(COLOR_WHEEL)//3], COLOR_WHEEL[2*len(COLOR_WHEEL)//3]]
            
        idx = COLOR_WHEEL.index(color)
        second_idx = (idx + len(COLOR_WHEEL) // 3) % len(COLOR_WHEEL)
        third_idx = (idx + 2 * len(COLOR_WHEEL) // 3) % len(COLOR_WHEEL)
        return [color, COLOR_WHEEL[second_idx], COLOR_WHEEL[third_idx]]

    def get_tetradic_colors(color):
        """Get four colors in a rectangle on the color wheel"""
        if color not in COLOR_WHEEL:
            return [COLOR_WHEEL[0], COLOR_WHEEL[2], COLOR_WHEEL[len(COLOR_WHEEL)//2], COLOR_WHEEL[len(COLOR_WHEEL)//2 + 2]]
            
        idx = COLOR_WHEEL.index(color)
        second_idx = (idx + 2) % len(COLOR_WHEEL)
        third_idx = (idx + len(COLOR_WHEEL) // 2) % len(COLOR_WHEEL)
        fourth_idx = (third_idx + 2) % len(COLOR_WHEEL)
        return [color, COLOR_WHEEL[second_idx], COLOR_WHEEL[third_idx], COLOR_WHEEL[fourth_idx]]

    def generate_dataset(samples_per_type=20):
        """Generate a dataset of color combinations"""
        dataset = []
        
        # 1. Monochromatic combinations (same color family for all)
        for color in COLORS:
            for _ in range(samples_per_type // len(COLORS)):
                light = random.choice(LIGHT_SHADES)
                dark = random.choice(DARK_SHADES)
                medium = random.choice(MEDIUM_SHADES)
                
                # Ensure all shades exist for this color
                if not all(color_exists(color, shade) for shade in [light, dark, medium, '200', '700', '600']):
                    continue
                    
                combination = {
                    "input": {
                        "strategy": "monochromatic",
                        "primary_color": color
                    },
                    "output": {
                        "primary-background": TAILWIND_COLORS[color][light],
                        "primary-text": TAILWIND_COLORS[color][dark],
                        "primary-accent": TAILWIND_COLORS[color][medium],
                        "secondary-background": TAILWIND_COLORS[color]['200'],
                        "secondary-text": TAILWIND_COLORS[color]['700'],
                        "secondary-accent": TAILWIND_COLORS[color]['600']
                    }
                }
                dataset.append(combination)
        
        # 2. Neutral with accent color
        for neutral in NEUTRALS:
            for accent in COLORS:
                # Ensure all required shades exist
                if not all(color_exists(neutral, shade) for shade in ['50', '100', '800', '900']) or \
                not all(color_exists(accent, shade) for shade in ['500', '600']):
                    continue
                    
                combination = {
                    "input": {
                        "strategy": "neutral_with_accent",
                        "neutral_color": neutral,
                        "accent_color": accent
                    },
                    "output": {
                        "primary-background": TAILWIND_COLORS[neutral]['50'],
                        "primary-text": TAILWIND_COLORS[neutral]['900'],
                        "primary-accent": TAILWIND_COLORS[accent]['500'],
                        "secondary-background": TAILWIND_COLORS[neutral]['100'],
                        "secondary-text": TAILWIND_COLORS[neutral]['800'],
                        "secondary-accent": TAILWIND_COLORS[accent]['600']
                    }
                }
                dataset.append(combination)
        
        # 3. Analogous colors (adjacent on color wheel)
        for color in COLORS:
            analogous = get_analogous_colors(color)
            
            # Ensure all required shades exist
            if not all(color_exists(color, shade) for shade in ['50', '100', '800', '900']) or \
            not color_exists(analogous[0], '500') or not color_exists(analogous[1], '600'):
                continue
                
            combination = {
                "input": {
                    "strategy": "analogous",
                    "primary_color": color,
                    "analogous_colors": analogous
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[color]['50'],
                    "primary-text": TAILWIND_COLORS[color]['900'],
                    "primary-accent": TAILWIND_COLORS[analogous[0]]['500'],
                    "secondary-background": TAILWIND_COLORS[color]['100'],
                    "secondary-text": TAILWIND_COLORS[color]['800'],
                    "secondary-accent": TAILWIND_COLORS[analogous[1]]['600']
                }
            }
            dataset.append(combination)
        
        # 4. Complementary colors
        for color in COLORS:
            complement = get_complementary_color(color)
            
            # Ensure all required shades exist
            if not all(color_exists(color, shade) for shade in ['50', '100', '800', '900']) or \
            not all(color_exists(complement, shade) for shade in ['500', '600']):
                continue
                
            combination = {
                "input": {
                    "strategy": "complementary",
                    "primary_color": color,
                    "complementary_color": complement
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[color]['50'],
                    "primary-text": TAILWIND_COLORS[color]['900'],
                    "primary-accent": TAILWIND_COLORS[complement]['500'],
                    "secondary-background": TAILWIND_COLORS[color]['100'],
                    "secondary-text": TAILWIND_COLORS[color]['800'],
                    "secondary-accent": TAILWIND_COLORS[complement]['600']
                }
            }
            dataset.append(combination)
        
        # 5. Triadic colors
        for color in COLORS:
            triadic = get_triadic_colors(color)
            
            # Ensure all required shades exist
            if not all(color_exists(triadic[0], shade) for shade in ['50', '100', '800', '900']) or \
            not color_exists(triadic[1], '500') or not color_exists(triadic[2], '600'):
                continue
                
            combination = {
                "input": {
                    "strategy": "triadic",
                    "colors": triadic
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[triadic[0]]['50'],
                    "primary-text": TAILWIND_COLORS[triadic[0]]['900'],
                    "primary-accent": TAILWIND_COLORS[triadic[1]]['500'],
                    "secondary-background": TAILWIND_COLORS[triadic[0]]['100'],
                    "secondary-text": TAILWIND_COLORS[triadic[0]]['800'],
                    "secondary-accent": TAILWIND_COLORS[triadic[2]]['600']
                }
            }
            dataset.append(combination)
        
        # 6. Tetradic/Rectangle colors
        for color in COLORS:
            tetradic = get_tetradic_colors(color)
            
            # Ensure all required shades exist
            if not all(color_exists(tetradic[0], shade) for shade in ['50', '900']) or \
            not color_exists(tetradic[1], '500') or \
            not all(color_exists(tetradic[2], shade) for shade in ['100', '800']) or \
            not color_exists(tetradic[3], '600'):
                continue
                
            combination = {
                "input": {
                    "strategy": "tetradic",
                    "colors": tetradic
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[tetradic[0]]['50'],
                    "primary-text": TAILWIND_COLORS[tetradic[0]]['900'],
                    "primary-accent": TAILWIND_COLORS[tetradic[1]]['500'],
                    "secondary-background": TAILWIND_COLORS[tetradic[2]]['100'],
                    "secondary-text": TAILWIND_COLORS[tetradic[2]]['800'],
                    "secondary-accent": TAILWIND_COLORS[tetradic[3]]['600']
                }
            }
            dataset.append(combination)
        
        # 7. Split complementary
        for color in COLORS:
            complement = get_complementary_color(color)
            complement_analogous = get_analogous_colors(complement)
            
            # Ensure all required shades exist
            if not all(color_exists(color, shade) for shade in ['50', '100', '800', '900']) or \
            not color_exists(complement_analogous[0], '500') or not color_exists(complement_analogous[1], '600'):
                continue
                
            combination = {
                "input": {
                    "strategy": "split_complementary",
                    "primary_color": color,
                    "complement_analogous": complement_analogous
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[color]['50'],
                    "primary-text": TAILWIND_COLORS[color]['900'],
                    "primary-accent": TAILWIND_COLORS[complement_analogous[0]]['500'],
                    "secondary-background": TAILWIND_COLORS[color]['100'],
                    "secondary-text": TAILWIND_COLORS[color]['800'],
                    "secondary-accent": TAILWIND_COLORS[complement_analogous[1]]['600']
                }
            }
            dataset.append(combination)
        
        # 8. Colored background with dark text
        for color in COLORS:
            light = random.choice(LIGHT_SHADES)
            
            # Ensure all required shades exist
            if not all(color_exists(color, shade) for shade in [light, '200', '600', '700']) or \
            not all(color_exists('gray', shade) for shade in ['800', '900']):
                continue
                
            combination = {
                "input": {
                    "strategy": "colored_background",
                    "background_color": color
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[color][light],
                    "primary-text": TAILWIND_COLORS['gray']['900'],
                    "primary-accent": TAILWIND_COLORS[color]['600'],
                    "secondary-background": TAILWIND_COLORS[color]['200'],
                    "secondary-text": TAILWIND_COLORS['gray']['800'],
                    "secondary-accent": TAILWIND_COLORS[color]['700']
                }
            }
            dataset.append(combination)
        
        # 9. Dark mode (dark background with light text)
        for color in COLORS + NEUTRALS:
            dark = random.choice(DARK_SHADES)
            accent = random.choice(COLORS)
            
            # Ensure all required shades exist
            if not all(color_exists(color, shade) for shade in [dark, '50', '100', '800']) or \
            not all(color_exists(accent, shade) for shade in ['400', '500']):
                continue
                
            combination = {
                "input": {
                    "strategy": "dark_mode",
                    "background_color": color,
                    "accent_color": accent
                },
                "output": {
                    "primary-background": TAILWIND_COLORS[color][dark],
                    "primary-text": TAILWIND_COLORS[color]['50'],
                    "primary-accent": TAILWIND_COLORS[accent]['400'],
                    "secondary-background": TAILWIND_COLORS[color]['800'],
                    "secondary-text": TAILWIND_COLORS[color]['100'],
                    "secondary-accent": TAILWIND_COLORS[accent]['500']
                }
            }
            dataset.append(combination)
        
        # 10. Pastel theme
        pastel_colors = [c for c in ['sky', 'pink', 'purple', 'green', 'yellow', 'orange'] if c in TAILWIND_COLORS]
        for primary in pastel_colors:
            for accent in pastel_colors:
                if primary != accent:
                    # Ensure all required shades exist
                    if not all(color_exists(primary, shade) for shade in ['50', '100']) or \
                    not all(color_exists(accent, shade) for shade in ['300', '400']) or \
                    not all(color_exists('gray', shade) for shade in ['700', '800']):
                        continue
                        
                    combination = {
                        "input": {
                            "strategy": "pastel",
                            "primary_color": primary,
                            "accent_color": accent
                        },
                        "output": {
                            "primary-background": TAILWIND_COLORS[primary]['100'],
                            "primary-text": TAILWIND_COLORS['gray']['800'],
                            "primary-accent": TAILWIND_COLORS[accent]['300'],
                            "secondary-background": TAILWIND_COLORS[primary]['50'],
                            "secondary-text": TAILWIND_COLORS['gray']['700'],
                            "secondary-accent": TAILWIND_COLORS[accent]['400']
                        }
                    }
                    dataset.append(combination)
        
        # 11. Professional/Corporate
        corporate_primaries = [c for c in ['blue', 'indigo', 'slate', 'gray'] if c in TAILWIND_COLORS]
        corporate_accents = [c for c in ['sky', 'blue', 'indigo', 'emerald', 'teal'] if c in TAILWIND_COLORS]
        
        for primary in corporate_primaries:
            for accent in corporate_accents:
                # Ensure all required shades exist
                if not all(color_exists(primary, shade) for shade in ['50', '100', '800', '900']) or \
                not all(color_exists(accent, shade) for shade in ['500', '600']):
                    continue
                    
                combination = {
                    "input": {
                        "strategy": "corporate",
                        "primary_color": primary,
                        "accent_color": accent
                    },
                    "output": {
                        "primary-background": TAILWIND_COLORS[primary]['50'],
                        "primary-text": TAILWIND_COLORS[primary]['900'],
                        "primary-accent": TAILWIND_COLORS[accent]['600'],
                        "secondary-background": TAILWIND_COLORS[primary]['100'],
                        "secondary-text": TAILWIND_COLORS[primary]['800'],
                        "secondary-accent": TAILWIND_COLORS[accent]['500']
                    }
                }
                dataset.append(combination)
        
        # 12. Vibrant/Creative
        vibrant_primaries = [c for c in ['purple', 'pink', 'rose', 'orange', 'amber', 'emerald'] if c in TAILWIND_COLORS]
        vibrant_accents = [c for c in ['yellow', 'lime', 'cyan', 'blue', 'fuchsia'] if c in TAILWIND_COLORS]
        
        for primary in vibrant_primaries:
            for accent in vibrant_accents:
                # Ensure all required shades exist
                if not all(color_exists(primary, shade) for shade in ['100', '200']) or \
                not all(color_exists(accent, shade) for shade in ['500', '600']) or \
                not all(color_exists('gray', shade) for shade in ['800', '900']):
                    continue
                    
                combination = {
                    "input": {
                        "strategy": "vibrant",
                        "primary_color": primary,
                        "accent_color": accent
                    },
                    "output": {
                        "primary-background": TAILWIND_COLORS[primary]['100'],
                        "primary-text": TAILWIND_COLORS['gray']['900'],
                        "primary-accent": TAILWIND_COLORS[accent]['500'],
                        "secondary-background": TAILWIND_COLORS[primary]['200'],
                        "secondary-text": TAILWIND_COLORS['gray']['800'],
                        "secondary-accent": TAILWIND_COLORS[accent]['600']
                    }
                }
                dataset.append(combination)
        
        return dataset

    # Generate the dataset and save to JSON
    dataset = generate_dataset(sample_size)  # Increased sample size
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} color combinations")
