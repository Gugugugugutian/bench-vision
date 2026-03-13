from datasets import load_dataset
from tqdm import tqdm
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./00_data")
args = parser.parse_args()

output_dir = args.output_dir
images_dir = os.path.join(output_dir, "images")
os.makedirs(output_dir, exist_ok=True)

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/RealWorldQA")

# Create directory for images
os.makedirs(images_dir, exist_ok=True)

# Convert to JSONL format
output_file = os.path.join(output_dir, "realworld_qa.jsonl")
image_count = 0

with open(output_file, 'w') as f:
    for split in dataset:
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            # Handle image field if it exists
            if 'image' in example and example['image'] is not None:
                image_path = os.path.join(images_dir, f"realworld_qa_{split}_{image_count}.png")
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
