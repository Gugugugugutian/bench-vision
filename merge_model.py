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
    device_map=None,
)

# Load LoRA
model = PeftModel.from_pretrained(base_model, lora_model_name)

# Pre-merge sanity check
processor = MllamaProcessor.from_pretrained(
    base_model_name,
    trust_remote_code=True
)
model.eval()
with torch.no_grad():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
            ],
        }
    ]
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        text=input_text,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=32,
    )
    gen_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    response_text = processor.batch_decode(
        gen_ids, skip_special_tokens=True
    )[0].strip()
    print(f"Pre-merge check output: {response_text}")

# Merge LoRA
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained(
    save_path,
    safe_serialization=True
)

# Save processor
processor.save_pretrained(save_path)

print(f"Model merged and saved to {save_path}")
