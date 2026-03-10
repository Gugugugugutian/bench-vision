from judger import Llama3Judger
from utils import load_file, save_file

import argparse
import os
import glob
import re

# Call method: 
# python evaluation.py \
#   --pred_folder ./predictions 
#   --output_folder ./evaluations

def evaluate(pred_folder, output_folder, dataset='all'):
    judger = Llama3Judger()

    format = "*.csv" if dataset == 'all' else f"{dataset}_response.csv"
    pred_files = glob.glob(os.path.join(pred_folder, format))

    os.makedirs(output_folder, exist_ok=True)

    results = []
    for pred_file in pred_files:
        print(f"\033[92m>>> Evaluating {pred_file}...\033[0m")
        pred_data = load_file(pred_file)

        step_results, step_score = judger.judge(pred_data)

        model, dataset = extract_model_and_dataset(pred_file)

        results.append({
            "model": model,
            "dataset": dataset,
            "step_score": step_score
        })
        save_file(
            step_results,
            os.path.join(output_folder, f"{os.path.basename(pred_file).replace('.jsonl', '_evaluation.csv')}")
        )

        os.makedirs(f"statistics/{model}", exist_ok=True)

        assert len(results) == 1, f"Only 1 file should be evaluated at a time for statistics. (got {len(results)} files)"
        save_file(results, f"statistics/{model}/{dataset}.csv")
        results = []  # reset results for next file

def extract_model_and_dataset(file_path: str): 
    m = re.search(r"predictions/([^/]+)/([^/_]+_[^/_]+)", file_path)
    model = m.group(1)
    dataset = m.group(2).replace(".csv", "")
    print(f"Model: {model}, Dataset: {dataset}")
    return model, dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions using Llama3Judger.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Folder containing prediction JSON files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save evaluation results.")
    parser.add_argument("--dataset", type=str, default='all', required=True, help="Name of the dataset.")
    args = parser.parse_args()
    
    evaluate(args.pred_folder, args.output_folder, dataset=args.dataset)