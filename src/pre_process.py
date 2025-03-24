import json
import re
import os
import glob
import random

def extract_css_vars_from_file(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error parsing JSON from {file_path}. Skipping.")
            return None
    
    # Extract data
    input_css = data.get("input", "")
    output_css = data.get("output", "")
    
    if not input_css or not output_css:
        print(f"Missing input or output in {file_path}. Skipping.")
        return None
    
    # Process input CSS with regex
    input_vars = {}
    pattern = r'--(\w+(?:-\w+)*): (\d+ \d+ \d+);'
    matches = re.findall(pattern, input_css)
    for var_name, value in matches:
        # Use the original variable name with dashes
        input_vars[var_name] = value
    
    # Process output CSS
    output_vars = {"root": {}, "dark": {}}
    
    # Regex pattern for root variables
    root_pattern = r':root\s*{([^}]*)}'
    root_match = re.search(root_pattern, output_css, re.DOTALL)
    if root_match:
        root_vars_text = root_match.group(1)
        var_pattern = r'--(\w+(?:-\w+)*): (\d+ \d+ \d+);'
        for var_name, value in re.findall(var_pattern, root_vars_text):
            output_vars["root"][var_name] = value
    
    # Regex pattern for dark mode variables
    dark_pattern = r'\.dark\s*{([^}]*)}'
    dark_match = re.search(dark_pattern, output_css, re.DOTALL)
    if dark_match:
        dark_vars_text = dark_match.group(1)
        var_pattern = r'--(\w+(?:-\w+)*): (\d+ \d+ \d+);'
        for var_name, value in re.findall(var_pattern, dark_vars_text):
            output_vars["dark"][var_name] = value
    
    # Create result structure for this file
    result = [{
        "input": {
            "primary-background": output_vars["root"]["primary-background"],
            "primary-text": output_vars["root"]["primary-text"],
            "primary-accent": output_vars["root"]["primary-accent"],
            "secondary-background": output_vars["root"]["secondary-background"],
            "secondary-text": output_vars["root"]["secondary-text"],
            "secondary-accent": output_vars["root"]["secondary-accent"],
        },
        "output": output_vars
    }, {
        "input": {
            "primary-background": output_vars["dark"]["primary-background"],
            "primary-text": output_vars["dark"]["primary-text"],
            "primary-accent": output_vars["dark"]["primary-accent"],
            "secondary-background": output_vars["dark"]["secondary-background"],
            "secondary-text": output_vars["dark"]["secondary-text"],
            "secondary-accent": output_vars["dark"]["secondary-accent"],
        },
        "output": output_vars
    }]
    
    return result

def pre_process_dataset_folder(colors_dataset_dir: str="dataset/color_palette_dataset", preprocessed_output: str="dataset/extracted_css_vars.json"):
    # Get all JSON files in the dataset folder
    json_files = glob.glob(f"{colors_dataset_dir}/*.json")
    
    # Process each file and collect results
    all_results = []
    for file_path in json_files:
        print(f"Processing {file_path}...")
        result = extract_css_vars_from_file(file_path)
        if result:
            all_results.extend(result)
    
    random.shuffle(all_results)
    # Write combined results to output file
    with open(preprocessed_output, 'w') as outfile:
        json.dump(all_results, outfile, indent=2)
    
    print(f"\nProcessing complete. Extracted data from {len(all_results)} files.")
    print(f"Output saved to {preprocessed_output}")

if __name__ == "__main__":
    pre_process_dataset_folder()
    