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
dataset = load_dataset("lmms-lab/textvqa", split="validation")

# Create directory for images
os.makedirs(images_dir, exist_ok=True)

# Convert to JSONL format
output_file = os.path.join(output_dir, "textvqa.jsonl")
image_count = 0
count = 0

def get_answer_by_majority_voting(answers: list) -> str:
    answer_counts = {}
    for ans in answers:
        ans_text = ans.strip()
        if ans_text in answer_counts:
            answer_counts[ans_text] += 1
        else:
            answer_counts[ans_text] = 1
    majority_answer = max(answer_counts, key=answer_counts.get)
    return majority_answer

with open(output_file, 'w') as f:
    split = "validation"
    for example in tqdm(dataset, desc=f"Processing {split}"):
        # Handle image field if it exists
        if 'image' in example and example['image'] is not None:
            image_path = os.path.join(images_dir, f"textvqa_{split}_{image_count}.png")

            image = example['image'].convert("RGB")  # Ensure image is in RGB format
            image.save(image_path)
            example['image'] = image_path
            image_count += 1

        f.write(json.dumps({
            'qid': f"textvqa_{split}{count:05d}",
            'question': example['question'],
            'solution': get_answer_by_majority_voting(example['answers']), 
            'image': example.get('image', "")
        }, ensure_ascii=False) + '\n')
        count += 1

print(f"Dataset converted and saved to {output_file}")
