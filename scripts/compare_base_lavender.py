#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def load_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"qid", "question", "answer", "solution", "score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {needed - set(df.columns)}")
    return df


def compare_dirs(base_dir: Path, lav_dir: Path) -> pd.DataFrame:
    rows = []
    base_files = {p.name: p for p in base_dir.glob("*.csv")}
    lav_files = {p.name: p for p in lav_dir.glob("*.csv")}
    common = sorted(set(base_files) & set(lav_files))

    for name in common:
        b = load_eval(base_files[name]).set_index("qid")
        l = load_eval(lav_files[name]).set_index("qid")
        common_qid = b.index.intersection(l.index)
        if common_qid.empty:
            continue
        b = b.loc[common_qid]
        l = l.loc[common_qid]
        diff = (b["score"] == 1) & (l["score"] == 0)
        if diff.any():
            sel_b = b[diff]
            sel_l = l[diff]
            for qid in sel_b.index:
                rows.append(
                    {
                        "dataset_file": name,
                        "qid": qid,
                        "question": sel_b.at[qid, "question"],
                        "solution": sel_b.at[qid, "solution"],
                        "base_answer": sel_b.at[qid, "answer"],
                        "base_score": sel_b.at[qid, "score"],
                        "lavender_answer": sel_l.at[qid, "answer"],
                        "lavender_score": sel_l.at[qid, "score"],
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Find cases where base model is correct but lavender is wrong."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Evaluation folder for base model (e.g. evaluations/Llama-3.2-11B-Vision-Instruct)",
    )
    parser.add_argument(
        "--lav_dir",
        type=str,
        required=True,
        help="Evaluation folder for lavender model (e.g. evaluations/Llama-3.2-11B-Vision-Instruct-lavender-official)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_vs_lavender_diff.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = compare_dirs(Path(args.base_dir), Path(args.lav_dir))
    if df.empty:
        print("No base-correct/lavender-wrong cases found.")
        return
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
