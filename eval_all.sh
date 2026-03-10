model_list=(
  "Llama-3.2-11B-Vision-Instruct"
  "Llama-3.2-11B-Vision-Instruct-lavender-official"
)

for model in "${model_list[@]}"; do
  model_path="../models/$model"

  if command -v nvidia-smi &> /dev/null; then
    echo -e "\033[32mGenerating responses for $model...\033[0m"
    python generate_response.py \
      --output_folder "./predictions/$model" \
      --model_path "$model_path" \
      --dataset "all"
  fi

  python evaluation.py \
    --pred_folder ./predictions/$model \
    --output_folder ./evaluations/$model \
    --dataset "all"
done
python summarize_statistics.py