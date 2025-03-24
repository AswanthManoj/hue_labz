COLOR_EXTRACTOR_INSTRUCTION = r'''# Color Palette Generation Prompt

## Task Description
Generate a complete color system for both light and dark themes based on the provided base colors. You will expand these base colors into a full set of CSS variables including hover, focus, disabled, border, and gradient variations. Apply professional color theory principles to ensure accessibility, harmony, and usability.

## Input Format
You will receive six base colors in RGB format:
```css
--primary-background: r g b;
--primary-text: r g b;
--primary-accent: r g b;
--secondary-background: r g b;
--secondary-text: r g b;
--secondary-accent: r g b;
```

## Output Requirements

### Light Theme Principles
- **Hover states**: Slightly darker/saturated than base (3-7% change)
- **Focus states**: More pronounced than hover (8-12% change from base)
- **Disabled states**: More muted, reduced opacity feel (15-25% toward background)
- **Border colors**: Often darker than base (15-20% shift)
- **Gradients**: Create a progression from base accent color:
    - gradient-a: Base color
    - gradient-b: Slight shift toward hover
    - gradient-c: Similar to focus
    - gradient-d: Deepest variant (15-20% shift from base)

### Dark Theme Principles
- **Create inverse theme**: If detecting light input colors, generate appropriate dark alternatives
- **Background colors**: Use dark variants (10-15 brightness or less)
- **Text colors**: Light variants (85-95 brightness)
- **Color relationships**: Maintain similar relationship patterns between elements
- **Contrast ratio**: Ensure WCAG AA compliance (4.5:1 for normal text)
- **Accent colors**: Brighter/more saturated in dark mode for visibility

### General Rules
- Maintain precise RGB format: `r g b` (space-separated integers)
- Preserve all variable names exactly as shown
- Ensure sufficient contrast between text and background colors
- Respect the original color's hue while adjusting saturation/brightness
- Accent colors should stand out but remain harmonious with the palette

### Output Format
Follow exactly this structure, replacing 'r g b' with appropriate RGB values:
```css
/* Light-mode Theme */
:root {
  /* Primary */
  --primary-background: r g b;
  --primary-background-hover: r g b;
  --primary-background-focus: r g b;
  --primary-background-disabled: r g b;
  --primary-background-border: r g b;

  --primary-text: r g b;
  --primary-text-hover: r g b;
  --primary-text-focus: r g b;
  --primary-text-disabled: r g b;
  --primary-text-border: r g b;

  --primary-accent: r g b;
  --primary-accent-hover: r g b;
  --primary-accent-focus: r g b;
  --primary-accent-disabled: r g b;
  --primary-accent-border: r g b;
  --primary-accent-gradient-a: r g b;
  --primary-accent-gradient-b: r g b;
  --primary-accent-gradient-c: r g b;
  --primary-accent-gradient-d: r g b;

  /* Secondary */
  --secondary-background: r g b;
  --secondary-background-hover: r g b;
  --secondary-background-focus: r g b;
  --secondary-background-disabled: r g b;
  --secondary-background-border: r g b;

  --secondary-text: r g b;
  --secondary-text-hover: r g b;
  --secondary-text-focus: r g b;
  --secondary-text-disabled: r g b;
  --secondary-text-border: r g b;

  --secondary-accent: r g b;
  --secondary-accent-hover: r g b;
  --secondary-accent-focus: r g b;
  --secondary-accent-disabled: r g b;
  --secondary-accent-border: r g b;
  --secondary-accent-gradient-a: r g b;
  --secondary-accent-gradient-b: r g b;
  --secondary-accent-gradient-c: r g b;
  --secondary-accent-gradient-d: r g b;
}

/* Dark-mode Theme */
.dark {
  /* Primary */
  --primary-background: r g b;
  --primary-background-hover: r g b;
  --primary-background-focus: r g b;
  --primary-background-disabled: r g b;
  --primary-background-border: r g b;

  --primary-text: r g b;
  --primary-text-hover: r g b;
  --primary-text-focus: r g b;
  --primary-text-disabled: r g b;
  --primary-text-border: r g b;

  --primary-accent: r g b;
  --primary-accent-hover: r g b;
  --primary-accent-focus: r g b;
  --primary-accent-disabled: r g b;
  --primary-accent-border: r g b;
  --primary-accent-gradient-a: r g b;
  --primary-accent-gradient-b: r g b;
  --primary-accent-gradient-c: r g b;
  --primary-accent-gradient-d: r g b;

  /* Secondary */
  --secondary-background: r g b;
  --secondary-background-hover: r g b;
  --secondary-background-focus: r g b;
  --secondary-background-disabled: r g b;
  --secondary-background-border: r g b;

  --secondary-text: r g b;
  --secondary-text-hover: r g b;
  --secondary-text-focus: r g b;
  --secondary-text-disabled: r g b;
  --secondary-text-border: r g b;

  --secondary-accent: r g b;
  --secondary-accent-hover: r g b;
  --secondary-accent-focus: r g b;
  --secondary-accent-disabled: r g b;
  --secondary-accent-border: r g b;
  --secondary-accent-gradient-a: r g b;
  --secondary-accent-gradient-b: r g b;
  --secondary-accent-gradient-c: r g b;
  --secondary-accent-gradient-d: r g b;
}
```

Always generate both light and dark themes, regardless of the original colors provided.
'''
