from datasets import load_dataset
from tqdm import tqdm
import json
import os

def process_single_text_question(question, options: list[str]) -> str:
    """
    Process a single text question and its options into a formatted string.
    
    Args:
        question (str): The question text.
        options (list[str]): A list of answer options.
        
    Returns:
        str: A formatted string containing the question and its options.
    """
    formatted_options = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
    return "Answer the question, only output the letter of your option. " + f"\n{question}\nOptions:\n{formatted_options}\nAnswer: "

# Download the RealworldQA dataset
dataset = load_dataset("lmms-lab/SuperGPQA")

# Create directory for images
os.makedirs("images", exist_ok=True)

# Convert to JSONL format
output_file = "supergpqa.jsonl"
count = 1
with open(output_file, 'w') as f:
    for split in dataset:
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            # Handle image field if it exists
            f.write(json.dumps({
                'qid': f"supergpqa_{split}{count:05d}",
                'question': process_single_text_question(example['question'], example['options']),
                'solution': example['answer_letter'],
                'image': ""
            }, ensure_ascii=False) + '\n')
            count += 1

print(f"Dataset converted and saved to {output_file}")

