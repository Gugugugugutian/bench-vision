base_path="/mnt/shared-storage-user/safevl-share_gpfs/gutian/bench-vision"

checkpoint=$1
checkpoint_path="$base_path/../models/$checkpoint-lora"

base_model_name=${2:-"Llama-3.2-11B-Vision-Instruct"}

lora_name="Ours-$checkpoint"

base_model_path="$base_path/../models/$base_model_name/"

echo ">>> Model path: $base_model_path"
echo ">>> Checkpoint path: $checkpoint_path"

bash_path="$base_path/merge_model.py"

job_name="merge-$checkpoint"

rjob submit \
  --name=$job_name \
  --gpu=2 \
  --memory=320000 \
  --cpu=32 \
  --charged-group=safevl_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/gutian-p:/mnt/shared-storage-user/gutian_gpfs \
  --mount=gpfs://gpfs1/safevl-share:/mnt/shared-storage-user/safevl-share_gpfs \
  --image=registry.h.pjlab.org.cn/ailab-safevl-safevl_gpu/llama:1 \
  -P 1 \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  -e DISABLE_P2P_CHECK=true \
  -e MASTER_PORT=29500 \
  -e PLANE_NO=1 \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  -- bash "$base_path/wrap_rjob.sh" "python $bash_path --lora_model_name $checkpoint_path --lora_name $lora_name"
