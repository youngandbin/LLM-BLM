import json
import os
from functools import reduce
from typing import Set

def get_keys_from_json(filepath: str) -> Set[str]:
    """Extract all keys from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        # If data is a dictionary, return its keys
        if isinstance(data, dict):
            return set(data.keys())
        # If data is a list of dictionaries, get unique keys from all dictionaries
        elif isinstance(data, list):
            all_keys = set()
            for item in data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            return all_keys
    return set()

def main():
    # Get all JSON files in the current directory
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the current directory")
        return
    
    # Get keys from first file
    all_keys = get_keys_from_json(json_files[0])
    
    # Find intersection with all other files
    for json_file in json_files[1:]:
        file_keys = get_keys_from_json(json_file)
        all_keys = all_keys.intersection(file_keys)
    
    print(f"Common keys across all {len(json_files)} JSON files:")
    for key in sorted(all_keys):
        print(f"- {key}")
    
    # save to common_stocks.json
    with open('./common_stocks.json', 'w') as f:
        json.dump(list(all_keys), f)

    # print length of common_stocks.json
    print(f"Length of common_stocks.json: {len(all_keys)}")

if __name__ == "__main__":
    main()
