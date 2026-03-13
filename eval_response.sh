#!/bin/bash
set -euo pipefail

# Usage:
#   bash eval_response.sh [model_name] [dataset]
# Example:
#   bash eval_response.sh Llama-3.2-11B-Vision-Instruct all

model_name=${1:-"all"}
dataset=${2:-"all"}

for mode in or llm str; do
  echo "[eval_response] running step5 mode=${mode}, model=${model_name}, dataset=${dataset}"
  bash step5.sh "$model_name" "$dataset" "$mode"
done

echo "[eval_response] completed modes: or, llm, str"
