#!/bin/bash
set -euo pipefail

model_name=${1:-"Llama-3.2-11B-Vision-Instruct"}
dataset=${2:-"all"}
judge_model=${3:-"../models/gpt-oss-20b"}
score_mode=${4:-"or"}

if [[ "$score_mode" == "or" ]]; then
  score_output_dir="./05_score_or"
elif [[ "$score_mode" == "llm" ]]; then
  score_output_dir="./07_score_llm"
elif [[ "$score_mode" == "str" ]]; then
  score_output_dir="./08_score_str"
else
  score_output_dir="./06_score_calibration"
fi

python pipeline/step4_evaluate_llm.py \
  --model_name "$model_name" \
  --dataset "$dataset" \
  --judge_model "$judge_model" \
  --response_dir "./01_response" \
  --output_dir "./03_llm_evaluation"

python pipeline/step5_score.py \
  --model_name "$model_name" \
  --dataset "$dataset" \
  --mode "$score_mode" \
  --str_eval_dir "./02_str_evaluation" \
  --llm_eval_dir "./03_llm_evaluation" \
  --calibration_dir "./04_calibration" \
  --output_dir "$score_output_dir"
