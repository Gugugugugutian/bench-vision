model=${1:-"Llama-3.2-11B-Vision-Instruct"}
model_path="../models/$model"
dataset=${2:-"all"}

if command -v nvidia-smi &> /dev/null; then
  echo -e "\033[32mGenerating responses for $model_path, dataset $dataset\033[0m"
  python generate_response.py \
    --output_folder "./predictions/$model" \
    --model_path "$model_path" \
    --dataset "$dataset"
fi

python evaluation.py \
  --pred_folder ./predictions/$model \
  --output_folder ./evaluations/$model \
  --dataset "$dataset"