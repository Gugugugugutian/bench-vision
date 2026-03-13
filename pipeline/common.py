from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_dataset_from_response_file(file_path: str | os.PathLike[str]) -> str:
    name = Path(file_path).name
    suffix = "_response.csv"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected response file name: {name}")
    return name[: -len(suffix)]


def list_dataset_jsonl(input_dir: str | os.PathLike[str], dataset: str = "all") -> list[Path]:
    base = Path(input_dir)
    if dataset == "all":
        files = sorted(base.glob("*.jsonl"))
    else:
        files = [base / f"{dataset}.jsonl"]
    return [p for p in files if p.exists()]


def list_response_csv(input_dir: str | os.PathLike[str], dataset: str = "all") -> list[Path]:
    base = Path(input_dir)
    if dataset == "all":
        files = sorted(base.glob("*_response.csv"))
    else:
        files = [base / f"{dataset}_response.csv"]
    return [p for p in files if p.exists()]


def extract_model_and_dataset_from_path(file_path: str | os.PathLike[str]) -> tuple[str, str]:
    p = Path(file_path)
    if p.parent.name:
        model = p.parent.name
    else:
        raise ValueError(f"Cannot infer model from path: {file_path}")

    dataset = parse_dataset_from_response_file(file_path)
    return model, dataset


def dataset_name_from_eval_file(file_path: str | os.PathLike[str]) -> str:
    name = Path(file_path).name
    # Keep compatibility with legacy naming.
    if name.endswith("_evaluation.csv"):
        return name[: -len("_evaluation.csv")]
    if name.endswith(".csv"):
        return name[:-4]
    raise ValueError(f"Unexpected evaluation file name: {name}")


def list_model_eval_files(root_dir: str | os.PathLike[str], model: str) -> dict[str, Path]:
    d = Path(root_dir) / model
    if not d.exists():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(d.glob("*.csv")):
        out[dataset_name_from_eval_file(p)] = p
    return out


def all_models_from_roots(*roots: str | os.PathLike[str]) -> list[str]:
    models = set()
    for root in roots:
        rd = Path(root)
        if not rd.is_dir():
            continue
        for child in rd.iterdir():
            if child.is_dir():
                models.add(child.name)
    return sorted(models)


def safe_key(model: str, qid: str) -> str:
    return f"{model}::{qid}"


def strtobool(text: str | None) -> bool:
    if text is None:
        return False
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}
