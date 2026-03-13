from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.common import ensure_dir, list_dataset_jsonl
from utils import load_file, save_file


def load_model(model_impl: str, model_path: str):
    if model_impl == "llama3":
        from models import Llama3

        return Llama3(model_path)
    raise ValueError(f"Unsupported --model_impl: {model_impl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Generate model responses into 01_response")
    parser.add_argument("--model_path", type=str, required=True, help="Path to tested model")
    parser.add_argument("--model_name", type=str, required=True, help="Name used for output subfolder")
    parser.add_argument("--model_impl", type=str, default="llama3", help="Model wrapper type")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or all")
    parser.add_argument("--input_dir", type=str, default="./00_data", help="Prepared data folder")
    parser.add_argument("--output_dir", type=str, default="./01_response", help="Root response output folder")
    args = parser.parse_args()

    overwrite = os.getenv("OVERWRITE", "").strip().lower() in {"1", "true", "yes"}

    model = load_model(args.model_impl, args.model_path)

    model_output_dir = ensure_dir(Path(args.output_dir) / args.model_name)
    input_files = list_dataset_jsonl(args.input_dir, dataset=args.dataset)
    if not input_files:
        raise FileNotFoundError(f"No input jsonl found in {args.input_dir} for dataset={args.dataset}")

    for input_file in input_files:
        dataset_name = input_file.stem
        out_path = model_output_dir / f"{dataset_name}_response.csv"
        if out_path.exists() and not overwrite:
            print(f"[step2] skip existing {out_path}")
            continue

        print(f"[step2] generating {input_file} -> {out_path}")
        input_data = load_file(str(input_file))
        response = model.predict(input_data)
        save_file(response, str(out_path))


if __name__ == "__main__":
    main()
