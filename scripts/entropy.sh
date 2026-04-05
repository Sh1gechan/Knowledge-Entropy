#!/bin/bash
#YBATCH -r a100_4
#SBATCH -N 1
#SBATCH -J ke-entropy
#SBATCH --time=6:00:00
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err

. /etc/profile.d/modules.sh
module load cuda/12.4

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

STEP=${1:-738020}
DATA_SIZE=${2:-2048}
BATCH_SIZE=${3:-4}

python -m analysis.entropy \
    --step "$STEP" \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE"

echo "Done"
