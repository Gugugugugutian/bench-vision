from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.common import all_models_from_roots, ensure_dir, list_model_eval_files


@dataclass
class VerifierConfig:
    enabled: bool
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class ExternalVerifier:
    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.client: Any = None
        self.client_mode: str | None = None

    def verify(self, question: str, answer: str, solution: str) -> tuple[float, str]:
        if not self.cfg.enabled:
            raise RuntimeError("External verifier is not enabled")
        if self.client is None:
            self._init_client()

        prompt = (
            "You are a strict evaluator for VQA/QA benchmark answers.\\n"
            "Decide if model answer is correct for the question, using reference solution.\\n"
            "Return ONLY YES or NO.\\n\\n"
            f"Question: {question}\\n"
            f"Model Answer: {answer}\\n"
            f"Reference Solution: {solution}\\n"
            "Final:"
        )

        text = self._chat_completion(prompt)
        verdict = text.lower()
        score = 1.0 if verdict.startswith("yes") else 0.0
        return score, text

    def _init_client(self) -> None:
        if not self.cfg.enabled:
            raise RuntimeError("External verifier is not enabled")
        if self.cfg.provider != "openai":
            raise ValueError(f"Unsupported verifier provider: {self.cfg.provider}")

        api_key = os.getenv(self.cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key env: {self.cfg.api_key_env}")

        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": api_key}
            if self.cfg.api_base:
                kwargs["base_url"] = self.cfg.api_base
            self.client = OpenAI(**kwargs)
            self.client_mode = "openai_v1"
            return
        except Exception:
            pass

        try:
            import openai  # type: ignore

            openai.api_key = api_key
            if self.cfg.api_base:
                openai.api_base = self.cfg.api_base
            self.client = openai
            self.client_mode = "openai_legacy"
            return
        except Exception as exc:
            raise RuntimeError(
                "openai package is required for calibrated mode with external verifier. Install/upgrade with: pip install -U openai"
            ) from exc

    def _chat_completion(self, prompt: str) -> str:
        if self.client_mode == "openai_v1":
            rsp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            return (rsp.choices[0].message.content or "").strip()

        if self.client_mode == "openai_legacy":
            rsp = self.client.ChatCompletion.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
            return (rsp["choices"][0]["message"]["content"] or "").strip()

        raise RuntimeError("External verifier client is not initialized")


def load_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"qid", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    keep = [c for c in ["qid", "question", "answer", "solution", "judge_output", "score"] if c in df.columns]
    df = df[keep].copy()
    df["qid"] = df["qid"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "qid",
                "final_score",
                "source",
                "verifier_model",
                "rule_score",
                "llm_score",
                "question",
                "answer",
                "solution",
                "verifier_output",
                "created_at_utc",
            ]
        )
    df = pd.read_csv(path)
    if "qid" in df.columns:
        df["qid"] = df["qid"].astype(str)
    return df


def save_cache(path: Path, cache_df: pd.DataFrame) -> None:
    cache_df = cache_df.drop_duplicates(subset=["dataset", "qid"], keep="last")
    cache_df = cache_df.sort_values(["dataset", "qid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(path, index=False)


def merge_for_dataset(rule_df: pd.DataFrame | None, llm_df: pd.DataFrame | None) -> pd.DataFrame:
    if rule_df is None and llm_df is None:
        return pd.DataFrame()
    if rule_df is None:
        merged = llm_df.copy()
        merged = merged.rename(columns={"score": "score_llm"})
        merged["score_rule"] = pd.NA
    elif llm_df is None:
        merged = rule_df.copy()
        merged = merged.rename(columns={"score": "score_rule"})
        merged["score_llm"] = pd.NA
    else:
        merged = pd.merge(rule_df, llm_df, on="qid", how="outer", suffixes=("_rule", "_llm"))

    for col in ["question", "answer", "solution"]:
        c_rule = f"{col}_rule"
        c_llm = f"{col}_llm"
        if c_rule in merged.columns or c_llm in merged.columns:
            merged[col] = merged[c_rule] if c_rule in merged.columns else pd.NA
            if c_llm in merged.columns:
                merged[col] = merged[col].fillna(merged[c_llm])

    if "score_rule" not in merged.columns:
        merged["score_rule"] = pd.NA
    if "score_llm" not in merged.columns:
        merged["score_llm"] = pd.NA

    merged["score_rule"] = pd.to_numeric(merged["score_rule"], errors="coerce")
    merged["score_llm"] = pd.to_numeric(merged["score_llm"], errors="coerce")
    merged["qid"] = merged["qid"].astype(str)
    return merged


def compute_or_score(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["score_final"] = out[["score_rule", "score_llm"]].fillna(0.0).max(axis=1)
    out["source"] = "or_max"
    return out


def compute_single_source_score(merged: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = merged.copy()
    if mode == "str":
        out["score_final"] = pd.to_numeric(out["score_rule"], errors="coerce").fillna(0.0)
        out["source"] = "str_direct"
    elif mode == "llm":
        out["score_final"] = pd.to_numeric(out["score_llm"], errors="coerce").fillna(0.0)
        out["source"] = "llm_direct"
    else:
        raise ValueError(f"Unsupported single-source mode: {mode}")
    return out


def compute_calibrated_score(
    merged: pd.DataFrame,
    model: str,
    dataset: str,
    cache_df: pd.DataFrame,
    verifier: ExternalVerifier,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    out = merged.copy()
    out["score_final"] = pd.NA
    out["source"] = ""

    cache_index = {}
    if not cache_df.empty:
        for _, row in cache_df.iterrows():
            cache_index[(str(row.get("dataset", "")), str(row["qid"]))] = row

    num_conflicts = 0
    num_api_calls = 0
    new_cache_rows: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        r = row.get("score_rule")
        l = row.get("score_llm")
        qid = str(row["qid"])

        r_missing = pd.isna(r)
        l_missing = pd.isna(l)

        if r_missing and l_missing:
            final = 0.0
            source = "missing_both"
        elif r_missing:
            final = float(l)
            source = "llm_only"
        elif l_missing:
            final = float(r)
            source = "rule_only"
        elif float(r) == float(l):
            final = float(r)
            source = "agree"
        else:
            num_conflicts += 1
            cache_key = (dataset, qid)
            cached = cache_index.get(cache_key)
            if cached is not None:
                final = float(cached["final_score"])
                source = "cache"
            else:
                final, verifier_output = verifier.verify(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    solution=str(row.get("solution", "")),
                )
                num_api_calls += 1
                source = "api_verify"
                new_cache_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "qid": qid,
                        "final_score": final,
                        "source": source,
                        "verifier_model": verifier.cfg.model,
                        "rule_score": float(r),
                        "llm_score": float(l),
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "solution": row.get("solution", ""),
                        "verifier_output": verifier_output,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

        out.at[idx, "score_final"] = float(final)
        out.at[idx, "source"] = source

    if new_cache_rows:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_cache_rows)], ignore_index=True)

    return out, cache_df, num_conflicts, num_api_calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: Merge evaluations and compute final score")
    parser.add_argument("--model_name", type=str, default="all", help="Model name or all")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name or all")
    parser.add_argument("--mode", choices=["or", "calibrated", "calibrate", "llm", "str"], default="or")
    parser.add_argument("--str_eval_dir", type=str, default="./02_str_evaluation")
    parser.add_argument("--llm_eval_dir", type=str, default="./03_llm_evaluation")
    parser.add_argument("--calibration_dir", type=str, default="./04_calibration")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--verifier_provider", type=str, default="openai")
    parser.add_argument("--verifier_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--verifier_api_base", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--verifier_api_key_env", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()

    if args.mode == "calibrate":
        args.mode = "calibrated"
    if not args.output_dir.strip():
        default_output = {
            "or": "./05_score_or",
            "calibrated": "./06_score_calibration",
            "llm": "./07_score_llm",
            "str": "./08_score_str",
        }
        args.output_dir = default_output[args.mode]

    if args.model_name == "all":
        models = all_models_from_roots(args.str_eval_dir, args.llm_eval_dir)
    else:
        models = [args.model_name]

    if not models:
        raise RuntimeError("No models found under evaluation roots")

    verifier = ExternalVerifier(
        VerifierConfig(
            enabled=args.mode == "calibrated",
            provider=args.verifier_provider,
            model=args.verifier_model,
            api_base=args.verifier_api_base or None,
            api_key_env=args.verifier_api_key_env,
        )
    )

    output_root = ensure_dir(args.output_dir)
    calibration_root = ensure_dir(args.calibration_dir)

    summary_rows: list[dict[str, Any]] = []

    for model in models:
        rule_files = list_model_eval_files(args.str_eval_dir, model)
        llm_files = list_model_eval_files(args.llm_eval_dir, model)
        datasets = sorted(set(rule_files) | set(llm_files))
        if args.dataset != "all":
            datasets = [d for d in datasets if d == args.dataset]

        if not datasets:
            print(f"[step5] skip model={model}: no datasets")
            continue

        model_out_dir = ensure_dir(output_root / model)
        cache_path = calibration_root / f"{model}.csv"
        cache_df = load_cache(cache_path)

        for dataset in datasets:
            rule_path = rule_files.get(dataset)
            llm_path = llm_files.get(dataset)

            rule_df = load_eval(rule_path) if rule_path else None
            llm_df = load_eval(llm_path) if llm_path else None
            merged = merge_for_dataset(rule_df, llm_df)
            if merged.empty:
                continue

            num_conflicts = 0
            num_api_calls = 0
            if args.mode == "or":
                num_conflicts = int(
                    (
                        merged["score_rule"].notna()
                        & merged["score_llm"].notna()
                        & (merged["score_rule"] != merged["score_llm"])
                    ).sum()
                )
                merged = compute_or_score(merged)
            elif args.mode in {"str", "llm"}:
                merged = compute_single_source_score(merged, mode=args.mode)
            else:
                merged, cache_df, num_conflicts, num_api_calls = compute_calibrated_score(
                    merged=merged,
                    model=model,
                    dataset=dataset,
                    cache_df=cache_df,
                    verifier=verifier,
                )

            out_cols = [
                "qid",
                "question",
                "answer",
                "solution",
                "score_rule",
                "score_llm",
                "score_final",
                "source",
            ]
            merged = merged[[c for c in out_cols if c in merged.columns]]

            out_path = model_out_dir / f"{dataset}_{args.mode}.csv"
            merged.to_csv(out_path, index=False)
            print(f"[step5] wrote {out_path}")

            step_score = float(pd.to_numeric(merged["score_final"], errors="coerce").fillna(0.0).mean())
            summary_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "mode": args.mode,
                    "step_score": step_score,
                    "num_samples": len(merged),
                    "num_conflicts": num_conflicts,
                    "num_api_calls": num_api_calls,
                }
            )

        if args.mode == "calibrated":
            save_cache(cache_path, cache_df)
            print(f"[step5] updated cache {cache_path}")

    if not summary_rows:
        raise RuntimeError("No scores were produced")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / f"summary_{args.mode}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[step5] wrote summary {summary_path}")


if __name__ == "__main__":
    main()
