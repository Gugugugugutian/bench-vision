from judger import Llama3Judger
from utils import load_file, save_file

import argparse
import os
import glob

# Call method: 
# python evaluation.py \
#   --pred_folder ./predictions 
#   --output_folder ./evaluations

def evaluate(pred_folder, output_folder):
    judger = Llama3Judger()
    pred_files = glob.glob(os.path.join(pred_folder, "*.csv"))

    os.makedirs(output_folder, exist_ok=True)

    results = []
    for pred_file in pred_files:
        print(f"\033[92m>>> Evaluating {pred_file}...\033[0m")
        pred_data = load_file(pred_file)

        step_results, step_score = judger.judge(pred_data)

        results.append({
            "file": pred_file,
            "step_score": step_score
        })
        save_file(
            step_results,
            os.path.join(output_folder, f"{os.path.basename(pred_file).replace('.jsonl', '_evaluation.csv')}")
        )

    save_file(results, "evaluation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions using Llama3Judger.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Folder containing prediction JSON files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save evaluation results.")
    args = parser.parse_args()
    
    evaluate(args.pred_folder, args.output_folder)