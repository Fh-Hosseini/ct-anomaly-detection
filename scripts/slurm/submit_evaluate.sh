#!/bin/bash
# Usage: bash scripts/slurm/submit_evaluate.sh <experiment_name>
EXPERIMENT_NAME_ARG="$1"
LOG_DIR="logs/resnet/${EXPERIMENT_NAME_ARG}/eval"
mkdir -p "$LOG_DIR"

sbatch --export=ALL,EVAL_EXPERIMENT_NAME=${EXPERIMENT_NAME_ARG} \
    --output=${LOG_DIR}/job%j.out --error=${LOG_DIR}/job%j.err \
    scripts/slurm/evaluate_resnet.sh