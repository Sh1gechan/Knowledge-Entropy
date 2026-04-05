#!/bin/bash
#YBATCH -r a100_4
#SBATCH -N 1
#SBATCH -J ke-resuscitation
#SBATCH --time=96:10:00
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err

# Resuscitation pipeline:
#   1. Compute Knowledge Entropy (generates mlp_average_coefficients.pt)
#   2. Modify parameters (generates resuscitation checkpoint)
#   3. Train with the resuscitated model
#
# Usage: ybatch scripts/resuscitation.sh [step] [ratio] [amplify] [config]

. /etc/profile.d/modules.sh
module load cuda/12.4

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

STEP=${1:-738020}
RATIO=${2:-0.5}
AMPLIFY=${3:-2.0}
CONFIG=${4:-configs/resuscitation/1B_bs128_lr4e4_pubmed_1ep_738k_resuscitation.yaml}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)}

CKPT_DIR="checkpoints/pretrained_1B/${STEP}-unsharded"
COEF_FILE="${CKPT_DIR}/mlp_average_coefficients.pt"
RESUS_FILE="${CKPT_DIR}/resuscitation_ratio${RATIO}_amplifying${AMPLIFY}.pt"

set -e

# Step 1: Knowledge Entropy (if not already computed)
if [ ! -f "$COEF_FILE" ]; then
    python -m analysis.entropy \
        --step "$STEP" \
        --data_size 2048 \
        --batch_size 4
fi

# Step 2: Change parameters (if not already done)
if [ ! -f "$RESUS_FILE" ]; then
    python -m analysis.change_parameters \
        --step "$STEP" \
        --resuscitation_ratio "$RATIO" \
        --amplifying_factor "$AMPLIFY"
fi

# Step 3: Train with resuscitated model
torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29599 \
    -m scripts.train "$CONFIG"

echo "Done"
