from datasets import load_dataset
from tqdm import tqdm
import json
import os

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/ai2d")

# Create directory for images
os.makedirs("images", exist_ok=True)

# Convert to JSONL format
output_file = "ai2d.jsonl"
image_count = 0
count = 0

def process(question: str, options: list[str], answer: int):
    prompt = f"Question: {question}\nOptions:\n"
    for i, option in enumerate(options):
        prompt += f"{i}. {option}\n"
    prompt += "Answer with the correct option number: "
    return prompt

with open(output_file, 'w') as f:
    for split in dataset:
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            # Handle image field if it exists
            if 'image' in example and example['image'] is not None:
                image_path = f"images/ai2d_{split}_{image_count}.png"
                example['image'].save(image_path)
                example['image'] = image_path
                image_count += 1
            f.write(json.dumps({
                'qid': f"ai2d_{split}{count:05d}",
                'question': process(example['question'], example['options'], example['answer']),
                'solution': example['answer'].strip(), 
                'image': example.get('image', "")
            }, ensure_ascii=False) + '\n')
            count += 1

print(f"Dataset converted and saved to {output_file}")