#!/bin/bash
PREPROCESSING_CONFIG_ARG="$1"
RECLIP_HU_MAX_ARG="$2"
if [ "$RECLIP_HU_MAX_ARG" == "none" ]; then
    RECLIP_HU_MAX_ARG=""
fi

NUM_JOBS_IN_CHAIN=4
SCRIPT=scripts/slurm/train_resnet.sh

EXPERIMENT_NAME=$(PREPROCESSING_CONFIG=$PREPROCESSING_CONFIG_ARG RECLIP_HU_MAX=$RECLIP_HU_MAX_ARG NEW_RUN=1 \
    python -c "from configs.resnet_baseline_config import EXPERIMENT_NAME; print(EXPERIMENT_NAME)")
echo "Claimed experiment name: $EXPERIMENT_NAME"

LOG_DIR="logs/resnet/${EXPERIMENT_NAME}/train"
mkdir -p "$LOG_DIR"

EXPORT_VARS="ALL,PREPROCESSING_CONFIG=${PREPROCESSING_CONFIG_ARG},RECLIP_HU_MAX=${RECLIP_HU_MAX_ARG},EXPERIMENT_NAME_OVERRIDE=${EXPERIMENT_NAME}"

first_job_output=$(sbatch --export=${EXPORT_VARS} --output=${LOG_DIR}/job%j.out --error=${LOG_DIR}/job%j.err $SCRIPT)
job_id=$(echo $first_job_output | awk '{print $4}')
echo "Submitted job 1/$NUM_JOBS_IN_CHAIN: $job_id"

for i in $(seq 2 $NUM_JOBS_IN_CHAIN); do
    next_job_output=$(sbatch --export=${EXPORT_VARS} --dependency=afterany:$job_id --output=${LOG_DIR}/job%j.out --error=${LOG_DIR}/job%j.err $SCRIPT)
    job_id=$(echo $next_job_output | awk '{print $4}')
    echo "Submitted job $i/$NUM_JOBS_IN_CHAIN: $job_id (depends on previous)"
done