from models import Llama3
from utils import load_file, save_file

import argparse
import os
import glob

# Call method:
# (Input from current directory)
# python generate_response.py \
#     --output_folder "./predictions" \
#     --model_path "path/to/llama3/model"

def load_model(model_path):
    model = Llama3(model_path)
    return model

def main(args):
    model = load_model(args.model_path)
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    # Process each input file
    input_files = glob.glob(os.path.join(args.input_folder, "*.jsonl"))
    for input_file in input_files:
        print(f"\033[92m>>> Processing file: {input_file}\033[0m")
        out_path = f"{args.output_folder}/{input_file.split('/')[-1].replace('.jsonl', '_response.csv')}"
        
        if os.path.exists(out_path):
            print(f"\033[93m>>> Output file already exists: {out_path}. Skipping...\033[0m")
            continue
        
        # Load input data
        input_data = load_file(input_file)
        # Generate response using the model
        response = model.predict(input_data)
        # Save the response to the output folder
        save_file(
            response, out_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using Llama3 model.")
    parser.add_argument("--input_folder", type=str, default=".", help="Folder containing input files.")
    parser.add_argument("--output_folder", type=str, default="./predictions", help="Folder to save generated responses.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Llama3 model.")
    
    args = parser.parse_args()
    main(args)