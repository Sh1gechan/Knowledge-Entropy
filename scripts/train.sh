#!/bin/bash
#YBATCH -r a100_8
#SBATCH -N 1
#SBATCH -J knowledge-entropy
#SBATCH --time=96:10:00
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

# Usage: ybatch scripts/train.sh <config_path>
# Example: ybatch scripts/train.sh configs/1B/1B_bs128_lr4e4_pubmed_1ep_738k.yaml

. /etc/profile.d/modules.sh
module load cuda/12.4

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

CONFIG=${1:-configs/1B/1B_bs128_lr4e4_pubmed_1ep_738k.yaml}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)}

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_JOB_GPUS (num: $NUM_GPUS)"
echo "Config: $CONFIG"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29599 \
    -m scripts.train "$CONFIG"

echo "Done"
