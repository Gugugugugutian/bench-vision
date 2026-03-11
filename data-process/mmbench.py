from datasets import load_dataset
from tqdm import tqdm
import json
import os

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/MMBench", 'en', split="dev")

# Create directory for images
os.makedirs("images", exist_ok=True)

# Convert to JSONL format
output_file = "mmbench.jsonl"
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
            image_path = f"images/mmbench_{split}_{image_count}.png"
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