from transformers import MllamaForConditionalGeneration, MllamaProcessor
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lora_model_name", default="../models/lavender-llama-3.2-11b-lora-official", type=str)
parser.add_argument("--lora_name", default="lavender-official", type=str)
args = parser.parse_args()

base_model_name = "../models/Llama-3.2-11B-Vision-Instruct"
lora_model_name = args.lora_model_name
lora_name = args.lora_name

save_path = f"../models/Llama-3.2-11B-Vision-Instruct-{lora_name}"

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