from datasets import load_dataset
from tqdm import tqdm
import json
import os

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/POPE", "default")

# Create directory for images
os.makedirs("images", exist_ok=True)

# Convert to JSONL format
output_file = "pope.jsonl"
image_count = 0
count = 0

with open(output_file, 'w') as f:
    for split in dataset:
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            # Handle image field if it exists
            if 'image' in example and example['image'] is not None:
                image_path = f"images/pope_{split}_{image_count}.png"
                example['image'].save(image_path)
                example['image'] = image_path
                image_count += 1
            f.write(json.dumps({
                'qid': f"pope_{split}{count:05d}",
                'question': example['question'],
                'solution': example['answer'].strip().split(',')[0],  # Take only the first line of the answer
                'image': example.get('image', "")
            }, ensure_ascii=False) + '\n')
            count += 1

print(f"Dataset converted and saved to {output_file}")