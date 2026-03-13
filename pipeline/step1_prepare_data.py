from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.common import ensure_dir, list_dataset_jsonl


DATA_PROCESS_SCRIPTS = {
    "ai2d": "ai2d.py",
    "mmbench": "mmbench.py",
    "pope": "pope.py",
    "realworld_qa": "realworld_qa.py",
    "supergpqa": "super_gpqa.py",
    "textvqa": "textvqa.py",
}


def copy_local_jsonl(input_dir: Path, output_dir: Path, dataset: str) -> None:
    files = list_dataset_jsonl(input_dir, dataset=dataset)
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {input_dir} for dataset={dataset}")
    for src in files:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        print(f"[step1] copied {src} -> {dst}")


def run_data_process_scripts(repo_root: Path, output_dir: Path, dataset: str) -> None:
    scripts_root = repo_root / "data-process"
    if not scripts_root.is_dir():
        raise FileNotFoundError(f"Missing folder: {scripts_root}")

    targets = DATA_PROCESS_SCRIPTS.keys() if dataset == "all" else [dataset]
    for ds in targets:
        script_name = DATA_PROCESS_SCRIPTS.get(ds)
        if not script_name:
            raise ValueError(f"Unsupported dataset for --mode process: {ds}")

        script_path = scripts_root / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing script: {script_path}")

        print(f"[step1] running {script_path} with --output_dir {output_dir}")
        subprocess.run(
            [sys.executable, str(script_path), "--output_dir", str(output_dir)],
            cwd=repo_root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Prepare benchmark data into 00_data")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or all")
    parser.add_argument(
        "--mode",
        choices=["copy", "process"],
        default="copy",
        help="copy: copy existing jsonl from --input_dir; process: run scripts in data-process",
    )
    parser.add_argument("--input_dir", type=str, default=".", help="Source dir for existing *.jsonl when mode=copy")
    parser.add_argument("--output_dir", type=str, default="./00_data", help="Prepared data output dir")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    repo_root = ROOT

    if args.mode == "copy":
        copy_local_jsonl(Path(args.input_dir), output_dir, dataset=args.dataset)
    else:
        run_data_process_scripts(repo_root, output_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
