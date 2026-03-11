model_list=(
  "Llama-3.2-11B-Vision-Instruct"
  "Llama-3.2-11B-Vision-Instruct-lavender-official"
  "Ours-online-diffusion"
  "Ours-distribution-1"
  "Ours-distribution-2"
)

dummy_model_name=$1
dataset_name=${2:-'all'}

for model in "${model_list[@]}"; do
  model_path="../models/$model"

  if command -v nvidia-smi &> /dev/null; then
    echo -e "\033[32mGenerating responses for $model...\033[0m"
    python generate_response.py \
      --output_folder "./predictions/$model" \
      --model_path "$model_path" \
      --dataset $dataset_name
  fi

  python evaluation.py \
    --pred_folder ./predictions/$model \
    --output_folder ./evaluations/$model \
    --dataset $dataset_name
done

if [ "$dataset_name" = "all" ]; then
  python summarize_statistics.py
fi