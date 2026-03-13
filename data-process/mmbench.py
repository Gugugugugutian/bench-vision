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
dataset = load_dataset("lmms-lab/MMBench", 'en', split="dev")

# Create directory for images
os.makedirs(images_dir, exist_ok=True)

# Convert to JSONL format
output_file = os.path.join(output_dir, "mmbench.jsonl")
image_count = 0
count = 0

def get_options(A, B, C=None, D=None):
    options = [A, B]
    if C != 'nan':
        options.append(C)
    if D != 'nan':
        options.append(D)
    return options

def process(question: str, options: list[str], answer: int):
    prompt = f"Question: {question}\nOptions:\n"
    prompt += "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
    prompt += "Answer the question, only output the letter of your option: "
    return prompt

with open(output_file, 'w') as f:
    split = "dev"
    for example in tqdm(dataset, desc=f"Processing {split}"):
        # Handle image field if it exists
        if 'image' in example and example['image'] is not None:
            image_path = os.path.join(images_dir, f"mmbench_{split}_{image_count}.png")
            example['image'].save(image_path)
            example['image'] = image_path
            image_count += 1
        options = get_options(example['A'], example['B'], example.get('C', 'nan'), example.get('D', 'nan'))
        f.write(json.dumps({
            'qid': f"mmbench_{split}{count:05d}",
            'question': process(example['question'], options, example['answer']),
            'solution': example['answer'].strip(), 
            'image': example.get('image', "")
        }, ensure_ascii=False) + '\n')
        count += 1

print(f"Dataset converted and saved to {output_file}")
