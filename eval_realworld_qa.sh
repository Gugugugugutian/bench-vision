model_list=(
  "Llama-3.2-11B-Vision-Instruct"
)

for model in "${model_list[@]}"; do
  echo "Evaluating $model on RealWorldQA..."
  model_path="../models/$model"

  if command -v nvidia-smi &> /dev/null; then
    python generate_response.py \
      --output_folder "./predictions/$model" \
      --model_path "$model_path"
  fi

  python evaluation.py \
    --pred_folder ./predictions/$model \
    --output_folder ./evaluations/$model
done