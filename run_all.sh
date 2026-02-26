#!/bin/bash
# Run all three datasets sequentially with 4-GPU distributed training.
# Usage: bash run_all.sh

set -e

NGPU=4
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  GAHR Full Reproduction - 3 Datasets"
echo "============================================"

for DATASET in rstp cuhk icfg; do
    CONFIG="${SCRIPT_DIR}/configs/${DATASET}.yaml"
    OUTPUT="${SCRIPT_DIR}/output/${DATASET}"
    LOG="${SCRIPT_DIR}/train_${DATASET}.log"

    echo ""
    echo "============================================"
    echo "  Training on ${DATASET} (config: ${CONFIG})"
    echo "  Output: ${OUTPUT}"
    echo "  Log: ${LOG}"
    echo "============================================"

    torchrun --nproc_per_node=${NGPU} \
        "${SCRIPT_DIR}/train.py" \
        --config "${CONFIG}" \
        --output_dir "${OUTPUT}" \
        2>&1 | tee "${LOG}"

    echo "  Done: ${DATASET}"
done

echo ""
echo "============================================"
echo "  All datasets complete!"
echo "============================================"
