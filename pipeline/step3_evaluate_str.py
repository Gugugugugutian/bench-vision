from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judger import Llama3Judger
from pipeline.common import ensure_dir, list_response_csv
from utils import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: String-match evaluation into 02_str_evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Model output subfolder name")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or all")
    parser.add_argument("--response_dir", type=str, default="./01_response", help="Root response folder")
    parser.add_argument("--output_dir", type=str, default="./02_str_evaluation", help="Root str-eval output folder")
    args = parser.parse_args()

    judger = Llama3Judger()

    model_response_dir = Path(args.response_dir) / args.model_name
    files = list_response_csv(model_response_dir, dataset=args.dataset)
    if not files:
        raise FileNotFoundError(f"No response files found in {model_response_dir}")

    out_dir = ensure_dir(Path(args.output_dir) / args.model_name)
    for pred_file in files:
        print(f"[step3] evaluating {pred_file}")
        pred_data = load_file(str(pred_file))
        step_results, step_score = judger.judge(pred_data)
        print(f"[step3] score={step_score:.4f} file={pred_file.name}")
        save_file(step_results, str(out_dir / pred_file.name))


if __name__ == "__main__":
    main()
