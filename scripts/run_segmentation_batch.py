"""
Segment a batch of CT volumes using Total Segmentator.

Called by scripts/slurm/segment_full.sh for each job in the array.

Args:
    start_index: first row index in volumes_path_list.csv (inclusive)
    end_index: last row index in volumes_path_list.csv (exclusive)  
    job_id: SLURM array task ID used for log file naming
"""


import sys
import time
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
VOLUMES_LIST_PATH = PROJECT_ROOT / "data/processed/volumes_path_list.csv"
LOG_DIR = Path("/anvme/workspace/iwi5437h-ct-anomaly-detection/logs/segmentation")

sys.path.insert(0, str(PROJECT_ROOT))
from src.ct_anomaly.data.segmentation import segment_one_volume


def is_segmented(mask_dir):
    """
    Check if the volume already segmented and 5 non empty lung lobes masks already exist in the directory. 

    Args:
        mask_dir: directory to the lung masks of a certain volume. 
    
    Returns:    
        a boolean value indicates if the masks already exist. 
    """

    if not mask_dir.exists():
        return False
    
    mask_files = list(mask_dir.glob("*.nii.gz"))

    # There should be exactly five non empty masks in the directory
    if len(mask_files) != 5:
        return False
    
    # Each of the mask files should have a non-zero size (non-empty)
    for mask_file in mask_files:
        if mask_file.stat().st_size <= 0:
            return False
    
    return True


def main():

    # get arguments: first and last index to segment and the job id
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    job_id = sys.argv[3]

    # stop 1 hour before SLURM's 24h limit
    job_start_time = time.time()
    MAX_JOB_SECONDS = 23 * 3600 

    # define log files for successful segmentaions and failed ones
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    segmented_log = LOG_DIR / f"segmented_job{job_id}.txt"
    not_segmented_log = LOG_DIR / f"not_segmented_job{job_id}.txt"

    volume_list_df = pd.read_csv(VOLUMES_LIST_PATH)
    segmenting_batch = volume_list_df.iloc[start_index: end_index]

    segmented_count = 0
    not_segmented_count = 0
    segmented_before_count = 0

    # start segmenting each volume in the batch
    for i, row in enumerate(segmenting_batch.itertuples()):

        # continue segmentation if we have at least more than 10 minutes to the maximum time 
        running_time = time.time() - job_start_time
        remaining_time = MAX_JOB_SECONDS - running_time

        if remaining_time < 600:  
            print(f"[STOP Job {job_id}] Only {remaining_time/60:.1f} minutes remaining")
            print(f"[Job {job_id}] Processed {i} volumes in this run, {len(segmenting_batch) - i} volumes are remaining")
            break


        volume_name = row.volume_name
        volume_path = Path(row.volume_data_path)
        mask_dir = Path(row.mask_dir)

        # skip if this volume already segmented before
        if is_segmented(mask_dir):
            segmented_before_count += 1
            print(f"Job {job_id} {i+1}-th volume in this batch already segmented before.")
            continue

        print(f"Job {job_id}: {i+1}-th volume processing: {volume_name}")

        try:
            segment_one_volume(
                volume_path = volume_path,
                mask_dir = mask_dir,
                device = "gpu"
            )

            with open(segmented_log, "a") as f:
                f.write(f"{volume_name}\n")

            segmented_count += 1
            print(f"Job {job_id}: {i+1}-th volume Done: {volume_name}")

        except Exception:
            with open(not_segmented_log, "a") as f:
                f.write(f"{volume_name}: \n{traceback.format_exc()}\n")

            not_segmented_count += 1
            print(f"Job {job_id}: {i+1}-th volume FAILED: {volume_name}")

        
    print()
    print(f"Job {job_id}: Finished segmenting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Job {job_id}: Success: {segmented_count} | Skipped(segmented before): {segmented_before_count} | Failed: {not_segmented_count}")


if __name__ == "__main__":
    main()