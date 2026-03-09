from datasets import load_dataset
from tqdm import tqdm
import json
import os

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/RealWorldQA")

# Create directory for images
os.makedirs("images", exist_ok=True)

# Convert to JSONL format
output_file = "realworld_qa.jsonl"
image_count = 0

with open(output_file, 'w') as f:
    for split in dataset:
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            # Handle image field if it exists
            if 'image' in example and example['image'] is not None:
                image_path = f"images/realworld_qa_{split}_{image_count}.png"
                example['image'].save(image_path)
                example['image'] = image_path
                image_count += 1
            f.write(json.dumps({
                'qid': f"realworld_{split}{image_count}",
                'question': example['question'],
                'solution': example['answer'],
                'image': example.get('image', "")
            }, ensure_ascii=False) + '\n')

print(f"Dataset converted and saved to {output_file}")