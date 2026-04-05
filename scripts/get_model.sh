#!/bin/bash
#
# Download OLMo-1B checkpoint from HuggingFace.
#
# Usage:
#   bash scripts/get_model.sh <step_number>
#   bash scripts/get_model.sh 738020
#
# The checkpoint is saved to:
#   checkpoints/pretrained_1B/<step_number>-unsharded/model.safetensors
#
# olmo/checkpoint.py automatically falls back to .safetensors when .pt is not found,
# so model.safetensors is sufficient — config.yaml and train.pt are not needed
# because the training config is provided via configs/ and optimizer state is reset
# (reset_optimizer_state: true in the yaml).

STEP="${1:-738020}"
REPO="allenai/OLMo-1B"

SCRIPT_DIR=$(dirname "$(realpath "$BASH_SOURCE")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
echo "Root dir: $ROOT_DIR"

# Resolve HuggingFace revision (branch) for the given step
REVISION=$(python3 -c "
from huggingface_hub import list_repo_refs
refs = list_repo_refs('${REPO}')
for b in refs.branches:
    if b.name.startswith('step${STEP}-'):
        print(b.name)
        break
" 2>/dev/null)

if [ -z "$REVISION" ]; then
    echo "ERROR: Could not find a HuggingFace branch for step ${STEP} in ${REPO}."
    echo "Available step branches:"
    python3 -c "
from huggingface_hub import list_repo_refs
refs = list_repo_refs('${REPO}')
for b in refs.branches:
    if b.name.startswith('step'):
        print(' ', b.name)
" 2>/dev/null
    exit 1
fi

echo "Resolved revision: ${REVISION}"

OUTPUT_PATH="${ROOT_DIR}/checkpoints/pretrained_1B/${STEP}-unsharded"
mkdir -p "$OUTPUT_PATH"

OUTPUT_FILE="${OUTPUT_PATH}/model.safetensors"
if [ -f "$OUTPUT_FILE" ]; then
    echo "File already exists: ${OUTPUT_FILE}. Skipping download."
    exit 0
fi

echo "Downloading model.safetensors -> ${OUTPUT_FILE}"
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='${REPO}',
    filename='model.safetensors',
    revision='${REVISION}',
    local_dir='${OUTPUT_PATH}',
)
print('Saved to:', path)
"
