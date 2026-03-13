#!/bin/bash
set -euo pipefail

model_name=${1:-"Llama-3.2-11B-Vision-Instruct"}
dataset=${2:-"all"}
model_path=${3:-"../models/$model_name"}

python pipeline/step2_generate_response.py \
  --model_name "$model_name" \
  --model_path "$model_path" \
  --dataset "$dataset" \
  --input_dir "./00_data" \
  --output_dir "./01_response"

python pipeline/step3_evaluate_str.py \
  --model_name "$model_name" \
  --dataset "$dataset" \
  --response_dir "./01_response" \
  --output_dir "./02_str_evaluation"
