#!/bin/bash
#SBATCH --job-name=resnet_baseline
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/resnet_baseline_%j.out
#SBATCH --error=logs/resnet_baseline_%j.err

cd ~/ct-anomaly-detection

source ~/.bashrc
conda activate ct-anomaly-detection

mkdir -p logs

python -u -m scripts.run_evaluate_resnet_baseline