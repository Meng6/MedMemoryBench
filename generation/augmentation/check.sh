#!/bin/bash
#   ./check.sh                        # Check all personas
#   ./check.sh --persona 1            # Check persona 1 only
#   ./check.sh --regenerate           # Enable regeneration
#   ./check.sh --dry-run              # Dry-run mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

DATASET_DIR="${PROJECT_ROOT}/generation/dataset"
OUTPUT_DIR="${PROJECT_ROOT}/generation/augmentation"

echo "=============================================="
echo "Query Difficulty Check"
echo "=============================================="
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Dataset directory: ${DATASET_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "----------------------------------------------"

python -m generation.augmentation.check.cli \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --persona 13 \
    "$@"
