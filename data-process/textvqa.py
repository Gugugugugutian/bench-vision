from datasets import load_dataset
from tqdm import tqdm
import json
import os

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/textvqa", split="validation")

# Create directory for images
os.makedirs("images", exist_ok=True)

# Convert to JSONL format
output_file = "textvqa.jsonl"
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
            image_path = f"images/textvqa_{split}_{image_count}.png"

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