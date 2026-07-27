#!/bin/bash

NUM_JOBS=4
SCRIPT=scripts/slurm/train_resnet.sh

# first_job_output=$(sbatch --export=ALL,NEW_RUN=1 $SCRIPT)
first_job_output=$(sbatch $SCRIPT)
job_id=$(echo $first_job_output | awk '{print $4}')
echo "Submitted job 1/$NUM_JOBS: $job_id"

for i in $(seq 2 $NUM_JOBS); do
    next_job_output=$(sbatch --dependency=afterany:$job_id $SCRIPT)
    job_id=$(echo $next_job_output | awk '{print $4}')
    echo "Submitted job $i/$NUM_JOBS: $job_id (depends on previous)"
done