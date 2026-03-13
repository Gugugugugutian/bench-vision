#!/bin/bash
set -euo pipefail

# Usage:
#   bash step3.sh [model_name] [dataset]

model_name=${1:-"Llama-3.2-11B-Vision-Instruct"}
dataset=${2:-"all"}

python pipeline/step3_evaluate_str.py \
  --model_name "$model_name" \
  --dataset "$dataset" \
  --response_dir "./01_response" \
  --output_dir "./02_str_evaluation"
