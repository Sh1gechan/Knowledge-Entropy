#!/bin/bash
#YBATCH -r epyc-7502_2
#SBATCH -N 1
#SBATCH -J prepare_pubmed
#SBATCH --time=96:10:00
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

. /etc/profile.d/modules.sh

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

python3 -u scripts/prepare_pubmed.py

echo "Done"
