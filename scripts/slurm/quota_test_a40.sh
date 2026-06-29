#!/bin/bash -l
#SBATCH --job-name=quota_a40
#SBATCH --output=/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/logs/quota_a40_%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/logs/quota_a40_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --array=0-29
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
sleep 300
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"