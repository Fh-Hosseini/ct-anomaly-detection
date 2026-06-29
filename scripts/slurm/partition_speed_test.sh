#!/bin/bash -l
#SBATCH --job-name=speed_test
#SBATCH --output=/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/logs/speed_test_a100_80_%j.out
#SBATCH --error=/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/logs/speed_test_a100_80_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=a100
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python/3.12-base
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ct-anomaly-detection

PROJECT_ROOT="/home/hpc/iwi5/iwi5437h/ct-anomaly-detection"

cd $PROJECT_ROOT

export PYTHONPATH=$PROJECT_ROOT

INPUT_PATH="/anvme/workspace/b180dc29-CT_RATE_IDEA_MIRROR/CT-RATE_train_fixed/train_4111/train_4111_a/train_4111_a_1.nii.gz"
OUTPUT_PATH="/anvme/workspace/iwi5437h-ct-anomaly-detection/lung_masks/speed_test_a100_80"

START_TIME=$(date +%s)
echo "Partition a100 (80GB)"

python scripts/run_segmentation.py \
    "train_4111_a_1" \
    "${INPUT_PATH}" \
    "${OUTPUT_PATH}"

END_TIME=$(date +%s)
RUN_TIME=$(( END_TIME - START_TIME ))

echo "Segmenting time: ${RUN_TIME} seconds"