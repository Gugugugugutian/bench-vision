from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judger import LLMJudger
from pipeline.common import ensure_dir, list_response_csv
from utils import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: LLM evaluation with vLLM into 03_llm_evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Tested model subfolder name")
    parser.add_argument("--judge_model", type=str, required=True, help="vLLM judge model path/name, e.g., ../models/gpt-oss-20b")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or all")
    parser.add_argument("--response_dir", type=str, default="./01_response", help="Root response folder")
    parser.add_argument("--output_dir", type=str, default="./03_llm_evaluation", help="Root llm-eval output folder")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=800)
    args = parser.parse_args()

    if LLMJudger is None:
        raise RuntimeError("LLMJudger is unavailable. Please install vllm and related dependencies.")

    judger = LLMJudger(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    model_response_dir = Path(args.response_dir) / args.model_name
    files = list_response_csv(model_response_dir, dataset=args.dataset)
    if not files:
        raise FileNotFoundError(f"No response files found in {model_response_dir}")

    out_dir = ensure_dir(Path(args.output_dir) / args.model_name)
    for pred_file in files:
        print(f"[step4] evaluating {pred_file}")
        pred_data = load_file(str(pred_file))
        step_results, step_score = judger.judge(pred_data)
        print(f"[step4] score={step_score:.4f} file={pred_file.name}")
        save_file(step_results, str(out_dir / pred_file.name))


if __name__ == "__main__":
    main()
