#!/bin/bash
set -euo pipefail

# Usage:
#   bash step5.sh [model_name] [dataset] [mode]
# Example:
#   bash step5.sh Llama-3.2-11B-Vision-Instruct all or
#   bash step5.sh Llama-3.2-11B-Vision-Instruct all calibrated

model_name=${1:-"all"}
dataset=${2:-"all"}
mode=${3:-"or"}

# ===== OpenAI calibration config (used only when mode=calibrated/calibrate) =====
# You can export these env vars before running, or modify defaults here.
OPENAI_API_KEY_VALUE=${OPENAI_API_KEY_VALUE:-""}
OPENAI_BASE_URL_VALUE=${OPENAI_BASE_URL_VALUE:-"https://api.openai.com/v1"}
OPENAI_VERIFY_MODEL=${OPENAI_VERIFY_MODEL:-"gpt-4.1-mini"}
OPENAI_API_KEY_ENV_NAME=${OPENAI_API_KEY_ENV_NAME:-"OPENAI_API_KEY"}

if [[ "$mode" == "or" ]]; then
  output_dir="./05_score_or"
elif [[ "$mode" == "calibrated" || "$mode" == "calibrate" ]]; then
  output_dir="./06_score_calibration"
elif [[ "$mode" == "llm" ]]; then
  output_dir="./07_score_llm"
elif [[ "$mode" == "str" ]]; then
  output_dir="./08_score_str"
else
  echo "ERROR: mode must be one of: or, calibrated, calibrate, llm, str"
  exit 1
fi

if [[ "$mode" == "calibrated" || "$mode" == "calibrate" ]]; then
  if [[ -n "$OPENAI_API_KEY_VALUE" ]]; then
    export "$OPENAI_API_KEY_ENV_NAME"="$OPENAI_API_KEY_VALUE"
  fi
  if [[ -z "${!OPENAI_API_KEY_ENV_NAME:-}" ]]; then
    echo "ERROR: ${OPENAI_API_KEY_ENV_NAME} is empty. Set OPENAI_API_KEY_VALUE or export ${OPENAI_API_KEY_ENV_NAME} first."
    exit 1
  fi

  python pipeline/step5_score.py \
    --model_name "$model_name" \
    --dataset "$dataset" \
    --mode "$mode" \
    --str_eval_dir "./02_str_evaluation" \
    --llm_eval_dir "./03_llm_evaluation" \
    --calibration_dir "./04_calibration" \
    --output_dir "$output_dir" \
    --verifier_provider "openai" \
    --verifier_model "$OPENAI_VERIFY_MODEL" \
    --verifier_api_base "$OPENAI_BASE_URL_VALUE" \
    --verifier_api_key_env "$OPENAI_API_KEY_ENV_NAME"
else
  python pipeline/step5_score.py \
    --model_name "$model_name" \
    --dataset "$dataset" \
    --mode "$mode" \
    --str_eval_dir "./02_str_evaluation" \
    --llm_eval_dir "./03_llm_evaluation" \
    --calibration_dir "./04_calibration" \
    --output_dir "$output_dir"
fi
