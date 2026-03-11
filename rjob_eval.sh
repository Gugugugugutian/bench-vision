# bash rjob_eval.sh realworld_qa [model_name]
# "Llama-3.2-11B-Vision-Instruct"
# "Llama-3.2-11B-Vision-Instruct-lavender-official"
base_path="/mnt/shared-storage-user/safevl-share_gpfs/gutian/bench-vision"

task=${1:-"all"}
model_name=${2:-"Llama-3.2-11B-Vision-Instruct"}

model_path="$base_path/../models/$model_name/"
# checkpoint_path="$base_path/../checkpoints/$checkpoint_name"
# echo "Evaluating checkpoint: $checkpoint_path"
echo ">>> Model path: $model_path"

if [ "$model_name" == "all" ]; then
  bash_path="$base_path/eval_all.sh"
else
  bash_path="$base_path/eval.sh"
fi

echo "Evaluation script path: $bash_path"
echo "Submitting evaluation job for task: $task"

job_name="eval-$task"

rjob submit \
  --name=$job_name \
  --gpu=1 \
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
  -- bash "$base_path/wrap_rjob.sh" $bash_path $model_name $task
