#!/bin/bash -l
#SBATCH --job-name=quota_a100_80
#SBATCH --output=/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/logs/quota_a100_80_%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5437h/ct-anomaly-detection/logs/quota_a100_80_%A_%a.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --partition=a100
#SBATCH --array=0-29
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

echo "Task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
sleep 300
echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"