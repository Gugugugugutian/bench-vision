from .BaseModel import BaseModel
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from tqdm import tqdm
from peft import PeftModel


class Llama3(BaseModel):
    def __init__(
            self, model_path
        ):
        super().__init__(
            model_path=model_path,
        )
    
    def load_model(self):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if self.checkpoint_path:
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
            self.model = self.model.merge_and_unload()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model.eval()
    
    def predict(
            self, 
            input_data: list[dict], 
            id_key: str = "qid",
            prompt_key: str = "question",
            response_key: str = "answer", 
            image_key: str = "image", 
            other_keys_to_keep: list = ['solution', ], 
        ) -> list:
        results = []
        for item in tqdm(input_data, desc="Llama3 Predicting"):
            image = None
            if image_key in item and item[image_key]:
                image_path = item[image_key]
                image = Image.open(image_path).convert("RGB")

            prompt_text = item[prompt_key]

            # Build conversation messages in Llama 3 vision-instruct format
            if image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            if image is not None:
                inputs = self.processor(
                    images=image,
                    text=input_text,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.processor(
                    text=input_text,
                    return_tensors="pt",
                ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )

            # Decode only the generated tokens (exclude input tokens)
            generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
            response_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            result = {
                id_key: item.get(id_key, None),
                prompt_key: prompt_text,
                response_key: response_text,
            }

            for key in other_keys_to_keep:
                if key in item:
                    result[key] = item[key]

            results.append(result)

        return results