import json
import os

def fix_results_json(filename="results/experiment_results.json"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        # Try to parse normally
        try:
            data = json.loads(content)
            print("JSON is valid.")
        except json.JSONDecodeError:
            print("JSON is corrupted, attempting to fix...")
            # Common fix: ensure it ends with }
            if not content.endswith('}'):
                # Try adding closing braces if it's a 'results' object
                if content.startswith('{'):
                    # Search for where results list might have cut off
                    if '"results": [' in content and ']' not in content[content.find('"results":'):]:
                        content += ']}'
                    else:
                        content += '}'
            
            data = json.loads(content)
            
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully repaired {filename}")
        
    except Exception as e:
        print(f"Failed to fix JSON: {e}")

if __name__ == "__main__":
    fix_results_json()
