#!/bin/bash -l
#SBATCH --job-name=preprocess_all
#SBATCH --output=/anvme/workspace/iwi5437h-ct-anomaly-detection/logs/preprocess_all/preprocess_all_%A_%a.out
#SBATCH --error=/anvme/workspace/iwi5437h-ct-anomaly-detection/logs/preprocess_all/preprocess_all_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --array=0-32%23
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python/3.12-base
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ct-anomaly-detection

# PATHS
PROJECT_ROOT="/home/hpc/iwi5/iwi5437h/ct-anomaly-detection"
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT

CONFIG_KEY="$1"

VOLUMES_PER_JOB=1500
START_IDX=$(( SLURM_ARRAY_TASK_ID * VOLUMES_PER_JOB ))
END_IDX=$(( START_IDX + VOLUMES_PER_JOB ))

echo "Job ${SLURM_ARRAY_TASK_ID}: Preprocessing volumes ${START_IDX} to ${END_IDX} at $(date)"
echo "Start: $(date)"

python scripts/run_preprocessing_batch.py \
    "${START_IDX}" \
    "${END_IDX}" \
    "${SLURM_ARRAY_TASK_ID}" \
    "${CONFIG_KEY}"

echo "End: $(date)"