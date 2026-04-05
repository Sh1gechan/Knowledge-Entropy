#!/bin/bash
#
# Generate or download the global_indices.npy file for OLMo-1B.
#
# global_indices.npy encodes the training data order used by the official OLMo-1B
# pretraining run, and is required by analysis/entropy.py (Knowledge Entropy calculation).
#
# The original URL (olmo-checkpoints.org) is no longer available for OLMo-1B.
# This script generates the file locally using the same deterministic logic as the
# original OLMo IterableDataset (no shuffle, sequential order), which matches the
# data_shuffling: false setting in configs/official/OLMo-1B.yaml.
#
# Requirements:
#   - .venv must be activated (source .venv/bin/activate)
#   - configs/official/OLMo-1B.yaml must exist
#   - The Dolma dataset paths in configs/official/OLMo-1B.yaml must be accessible

SCRIPT_DIR=$(dirname "$(realpath "$BASH_SOURCE")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
FILE_PATH="${ROOT_DIR}/data/global_indices/1B/global_indices.npy"

if [ -f "$FILE_PATH" ]; then
    echo "File already exists at: ${FILE_PATH}. Skipping."
    exit 0
fi

mkdir -p "$(dirname "$FILE_PATH")"

echo "Generating global_indices.npy locally..."
cd "$ROOT_DIR"

python3 - <<'EOF'
import os
import sys
import numpy as np
from pathlib import Path

output_path = "data/global_indices/1B/global_indices.npy"
config_path = "configs/official/OLMo-1B.yaml"

if not os.path.isfile(config_path):
    print(f"ERROR: {config_path} not found.")
    print("The official OLMo-1B config is required to determine the dataset size.")
    sys.exit(1)

try:
    from olmo.config import TrainConfig
    from olmo.data import build_memmap_dataset
except ImportError as e:
    print(f"ERROR: {e}")
    print("Make sure the venv is activated: source .venv/bin/activate")
    sys.exit(1)

print(f"Loading config from {config_path} ...")
cfg = TrainConfig.load(config_path)

print("Loading dataset to determine size (this may take a moment)...")
dataset = build_memmap_dataset(cfg, cfg.data)
n = len(dataset)
print(f"Dataset size: {n:,} samples")

# Replicate OLMo IterableDataset._build_global_indices with shuffle=False
# (data_shuffling: false in OLMo-1B.yaml)
indices = np.arange(n, dtype=np.uint32)

print(f"Saving to {output_path} ...")
mmap = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=(n,))
mmap[:] = indices
mmap.flush()
del mmap
print(f"Done. Saved {n:,} indices to {output_path}")
EOF
