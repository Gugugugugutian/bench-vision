#!/bin/bash
set -e

script_name="$1"
shift
script_args=("$@")

echo "Script Name: $script_name"
echo "Script Args: ${script_args[*]}"

echo "Running script: $script_name"
sleep 2

mv /root/miniconda3 /root/oldminiconda3
ln -fs /mnt/shared-storage-user/gutian_gpfs/miniconda3 /root/
ln -fs /mnt/shared-storage-user/gutian_gpfs /root/worker_gpfs
ln -fs /mnt/shared-storage-user/safevl-share_gpfs /root/team_gpfs
source /mnt/shared-storage-user/gutian_gpfs/miniconda3/etc/profile.d/conda.sh

# copy nltk data
cp -r /root/worker_gpfs/nltk_data/ /root

unset http_proxy
unset https_proxy
unset no_proxy

export NCCL_DEBUG=WARN
echo "NCCL environment variables have been set."

export CUDA_HOME=/mnt/shared-storage-user/gutian_gpfs/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME set to $CUDA_HOME"
echo "PATH updated to include CUDA binaries."

export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/"
export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"

echo "PIP index URLs and trusted hosts have been set."

cd /mnt/shared-storage-user/safevl-share_gpfs/gutian/bench-vision
if [[ "$script_name" == *"step4"* ]]; then
    conda activate vlm
else
    conda activate llama32
fi

# Verify the script exists
if [[ ! -f "$script_name" ]]; then
    bash -lc "$script_name"
    exit 0
fi

echo "============= Running the script with the provided arguments ============="
echo "Script: $script_name"
echo "Args: ${script_args[*]}"

if [[ -f "$script_name" ]]; then
    bash "$script_name" "${script_args[@]}"
else
    bash -lc "$script_name ${script_args[*]}"
fi
