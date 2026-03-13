#!/bin/bash
set -euo pipefail

# Usage: bash rjob_step2.sh [dataset] [model_name] [model_path]
base_path="/mnt/shared-storage-user/safevl-share_gpfs/gutian/bench-vision"

dataset=${1:-"all"}
model_name=${2:-"Llama-3.2-11B-Vision-Instruct"}
model_path=${3:-"$base_path/../models/$model_name"}

bash_path="$base_path/step2.sh"
job_name="step2-${model_name}-${dataset}"
job_name=${job_name//\//-}

echo "Submitting step2 job"
echo "dataset=$dataset model_name=$model_name model_path=$model_path"

rjob submit \
  --name="$job_name" \
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
  -- bash "$base_path/wrap_rjob.sh" "$bash_path" "$model_name" "$dataset" "$model_path"
