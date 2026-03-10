from transformers import MllamaForConditionalGeneration, MllamaProcessor
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_name", default="../models/Llama-3.2-11B-Vision-Instruct", type=str)
parser.add_argument("--lora_model_name", default="../models/lavender-llama-3.2-11b-lora-official", type=str)
parser.add_argument("--lora_name", default="Ours-new", type=str)
args = parser.parse_args()

base_model_name = args.base_model_name
lora_model_name = args.lora_model_name
lora_name = args.lora_name

save_path = f"../models/{lora_name}"

print(f"Base model: {base_model_name}")
print(f"LoRA model: {lora_model_name}")
print(f"Save path: {save_path}")

# Load base model
base_model = MllamaForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA
model = PeftModel.from_pretrained(base_model, lora_model_name)

# Merge LoRA
merged_model = model.merge_and_unload()

merged_model.save_pretrained(
    save_path,
    # max_shard_size="5GB"
)
processor = MllamaProcessor.from_pretrained(base_model_name)
processor.save_pretrained(save_path)

print(f"Model merged and saved to {save_path}")