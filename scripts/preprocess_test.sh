#!/bin/bash
#SBATCH --job-name=preprocess_test
#SBATCH --output=/anvme/workspace/iwi5437h-ct-anomaly-detection/logs/preprocess_test_%j.out
#SBATCH --error=/anvme/workspace/iwi5437h-ct-anomaly-detection/logs/preprocess_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:1

# Paths
PROJECT_DIR="/home/hpc/iwi5/iwi5437h/ct-anomaly-detection"
DATA_PATH="/anvme/workspace/b180dc29-CT_RATE_IDEA_MIRROR"
MASKS_PATH="/anvme/workspace/iwi5437h-ct-anomaly-detection/lung_masks"
HOME_MASKS="/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/data/processed/lung_masks"
OUTPUT_PATH="/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/data/processed/preprocessed"

# Environment setup
module load python/3.12-base
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ct-anomaly-detection
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR

# Test volumes
VOLUMES=(
    "valid_375/valid_375_a/valid_375_a_1.nii.gz"
    "train_6167/train_6167_a/train_6167_a_1.nii.gz"
    "train_4111/train_4111_a/train_4111_a_1.nii.gz"
    "train_19305/train_19305_a/train_19305_a_1.nii.gz"
    "train_16902/train_16902_a/train_16902_a_1.nii.gz"
)

# Run preprocessing for each volume
for VOLUME_PATH in "${VOLUMES[@]}"; do
    SPLIT=$(echo $VOLUME_PATH | cut -d'/' -f1 | cut -d'_' -f1)
    INPUT_PATH="${DATA_PATH}/CT-RATE_${SPLIT}_fixed/${VOLUME_PATH}"
    VOLUME_NAME=$(basename $VOLUME_PATH .nii.gz)
    OUTPUT_FILE="${OUTPUT_PATH}/${VOLUME_NAME}.npz"

    # Use home masks for train_4111_a_1, anvme for others
    if [ -d "${MASKS_PATH}/${VOLUME_NAME}" ]; then
        MASK_DIR="${MASKS_PATH}/${VOLUME_NAME}"
    else
        MASK_DIR="${HOME_MASKS}/${VOLUME_NAME}"
    fi

    python scripts/run_preprocessing.py \
        "${VOLUME_NAME}" \
        "${INPUT_PATH}" \
        "${MASK_DIR}" \
        "${OUTPUT_FILE}"
done

echo "All 5 volumes preprocessed successfully."