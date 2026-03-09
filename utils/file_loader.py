import json
import os 
import pandas as pd

def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_csv(file_path):
    """Load a CSV file and return a list of dictionaries."""
    return pd.read_csv(file_path).to_dict(orient='records')

def save_csv(data, file_path):
    """Save a pandas DataFrame to a CSV file."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data.to_csv(file_path, index=False)

def load_file(file_path):
    """Load a file based on its extension."""
    _, ext = os.path.splitext(file_path)
    if ext == '.jsonl':
        return load_jsonl(file_path)
    elif ext == '.csv':
        return load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
def save_file(data, file_path):
    """Save data to a file based on its extension."""
    _, ext = os.path.splitext(file_path)
    if ext == '.jsonl':
        save_jsonl(data, file_path)
    elif ext == '.csv':
        save_csv(data, file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")