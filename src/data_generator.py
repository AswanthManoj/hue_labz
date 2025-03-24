import os, time
import json, asyncio
from pathlib import Path
from src.generative import COLOR_EXTRACTOR_INSTRUCTION
from src.dependencies import LLMConnector, LLMProviderConfig
from src.color_compute import (monochromatic, neutral_with_accent, analogous, 
complementary, triadic, split_complementary, colored_background, dark_mode, pastel, corporate, vibrant)


llm = LLMConnector(
    providers=[
        LLMProviderConfig(
            provider="google",
            api_key=[os.getenv("GOOGLE_API_KEY")],
            primary_model="gemini-2.0-pro-exp-02-05",
            secondary_model="gemini-2.0-pro-exp-02-05",
            base_url="https://generativelanguage.googleapis.com/v1beta"
        )
    ],
    provider_priority=["google"]
)

strategy_map = {
    "monochromatic": monochromatic,
    "neutral_with_accent": neutral_with_accent,
    "analogous": analogous,
    "complementary": complementary,
    "triadic": triadic,
    "tetradic": triadic,
    "split_complementary": split_complementary,
    "colored_background": colored_background,
    "dark_mode": dark_mode,
    "pastel": pastel,
    "corporate": corporate,
    "vibrant": vibrant
}

async def generate_colorset(item):
    icl_example = strategy_map[item["input"]["strategy"]]
    input = icl_example["input"]
    output = icl_example["output"]
    new_input = item["output"]
    response = await llm.generate(
        messages=[{
            "role": "user",
            "content": f'''```css
{input}
```'''
        }, {
            "role": "assistant",
            "content": f'''```css
{output}
```'''
        }, {
            "role": "user",
            "content": f'''```css
--primary-background: {new_input['primary-background']};
--primary-text: {new_input['primary-text']};
--primary-accent: {new_input['primary-accent']};
--secondary-background: {new_input['secondary-background']};
--secondary-text: {new_input['secondary-text']};
--secondary-accent: {new_input['secondary-accent']};
```'''
        }],
        system=COLOR_EXTRACTOR_INSTRUCTION, 
        max_tokens=6000, 
        # enable_cache=True, 
        use_primary_model=True,
        temperature=0.4, provider="google"
    )
    # time.sleep(2)
    return response

async def generate_expanded_palette_dataset(color_pallet_dataset: str="dataset/color_palette_dataset.json", output_dir: str="dataset/color_palette_dataset"):
    # 1. Read the JSON file
    try:
        with open(color_pallet_dataset, "r") as f:
            data = json.load(f)
        print(f"Loaded dataset with {len(data)} items")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return
    
    # 2. Create dataset folder if it doesn't exist
    dataset_folder = Path(output_dir)
    dataset_folder.mkdir(exist_ok=True)
    
    # 3. Create/load progress tracking dictionary
    progress_file = dataset_folder / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
            start_index = progress["last_index"] + 1
            print(f"Resuming from index {start_index}")
        except Exception as e:
            print(f"Error reading progress file: {e}")
            progress = {"last_index": -1}
            start_index = 0
    else:
        progress = {"last_index": -1}
        start_index = 0
        print("Starting fresh processing")
    
    # 4 & 5. Process each item and save results
    for i in range(start_index, len(data)):
        item = data[i]
        
        processed_item = await generate_colorset(item)
        new_data = {
            "input": f'''```css
--primary-background: {item['output']['primary-background']};
--primary-text: {item['output']['primary-text']};
--primary-accent: {item['output']['primary-accent']};
--secondary-background: {item['output']['secondary-background']};
--secondary-text: {item['output']['secondary-text']};
--secondary-accent: {item['output']['secondary-accent']};
```''',
            "output": processed_item
        }

        output_path = dataset_folder / f"{i}_data.json"
        try:
            with open(output_path, "w") as f:
                json.dump(new_data, f, indent=2)
        except Exception as e:
            print(f"Error saving item {i}: {e}")
            continue
        
        # Update progress
        progress["last_index"] = i
        try:
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"Error updating progress: {e}")
        
        print(f"Processed item {i} of {len(data)}")

if __name__ == "__main__":
    asyncio.run(generate_expanded_palette_dataset())
