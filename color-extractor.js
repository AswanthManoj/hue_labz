const fs = require('fs');
const colors = require('tailwindcss/colors');
const culori = require('culori');

// Function to convert any color format to our standardized output
function convertColor(colorValue) {
  try {
    // Parse the color using culori (handles hex, rgb, hsl, lab, lch, oklch, etc.)
    const parsed = culori.parse(colorValue);
    
    if (!parsed) {
      console.warn(`Warning: Could not parse color: ${colorValue}`);
      return {
        original: colorValue,
        hex: '#000000',
        rgb: '0 0 0',
        rgbArray: [0, 0, 0]
      };
    }

    // Convert to RGB object
    const rgbObj = culori.rgb(parsed);
    
    // Round and scale RGB values to 0-255 range
    const r = Math.round(rgbObj.r * 255);
    const g = Math.round(rgbObj.g * 255);
    const b = Math.round(rgbObj.b * 255);
    
    // Generate hex string
    const hexColor = culori.formatHex(rgbObj);
    
    return {
      original: colorValue,
      hex: hexColor,
      rgb: `${r} ${g} ${b}`,
      rgbArray: [r, g, b]
    };
  } catch (e) {
    console.error(`Error converting color: ${colorValue}`, e);
    return {
      original: colorValue,
      hex: '#000000',
      rgb: '0 0 0',
      rgbArray: [0, 0, 0]
    };
  }
}

// Color palettes we want to include from Tailwind
const colorNames = [
  'red', 'orange', 'amber', 'yellow', 'lime', 'green', 'emerald', 
  'teal', 'cyan', 'sky', 'blue', 'indigo', 'violet', 'purple', 
  'fuchsia', 'pink', 'rose', 'slate', 'gray', 'zinc', 'neutral', 'stone'
];

// Shades we want to include
const shades = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

// Create the tailwind colors object
const tailwindColors = {};

// Add fallback mechanism - if a color/shade doesn't exist in latest, use v2 colors
const fallbackColors = {
  // Defining a few common fallbacks just in case
  violet: {
    50: '#f5f3ff', 100: '#ede9fe', 200: '#ddd6fe', 300: '#c4b5fd',
    400: '#a78bfa', 500: '#8b5cf6', 600: '#7c3aed', 700: '#6d28d9',
    800: '#5b21b6', 900: '#4c1d95', 950: '#2e1065'
  }
};

colorNames.forEach(colorName => {
  // Skip colors that don't exist in the official palette
  if (!colors[colorName] && !fallbackColors[colorName]) {
    console.warn(`Warning: Color '${colorName}' not found in Tailwind palette`);
    return;
  }
  
  const colorSource = colors[colorName] || fallbackColors[colorName];
  tailwindColors[colorName] = {};
  
  shades.forEach(shade => {
    // Skip shades that don't exist for this color
    if (!colorSource[shade]) {
      console.warn(`Warning: Shade ${shade} not found for color '${colorName}'`);
      return;
    }
    
    tailwindColors[colorName][shade] = convertColor(colorSource[shade]);
  });
});

// Save to file
const jsonData = JSON.stringify(tailwindColors, null, 2);
fs.writeFileSync('dataset/tailwind-colors.json', jsonData);

console.log('Tailwind colors saved to tailwind-colors.json');

// Generate Python-friendly format (just the RGB values as space-separated strings)
const pythonFormatColors = {};
Object.entries(tailwindColors).forEach(([colorName, shades]) => {
  pythonFormatColors[colorName] = {};
  Object.entries(shades).forEach(([shade, colorData]) => {
    pythonFormatColors[colorName][shade] = colorData.rgb;
  });
});

fs.writeFileSync('dataset/tailwind-colors-python-format.json', 
  JSON.stringify(pythonFormatColors, null, 2));
console.log('Python-friendly format saved to tailwind-colors-python-format.json');

// Also create a CSS variables file for reference
let cssVars = ":root {\n";
Object.entries(tailwindColors).forEach(([colorName, shades]) => {
  Object.entries(shades).forEach(([shade, colorData]) => {
    cssVars += `  --${colorName}-${shade}: ${colorData.rgb};\n`;
  });
});
cssVars += "}\n";

fs.writeFileSync('dataset/tailwind-colors.css', cssVars);
console.log('CSS variables saved to tailwind-colors.css');

// Display a sample for verification
console.log('\nSample colors for verification:');
console.log('Blue 500:', tailwindColors.blue[500]);
console.log('Red 500:', tailwindColors.red[500]);
console.log('Emerald 500:', tailwindColors.emerald[500]);
