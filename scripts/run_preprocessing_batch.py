import sys
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.ct_anomaly.data.preprocessing import preprocess_one_volume
from configs.preprocessing_configs import CONFIGS

 # stop 1 hour before SLURM's 24h limit
MAX_JOB_SECONDS = 23 * 3600 


def is_preprocessed(preprocessed_path):
    """
    Check if the volume already preprocessed.

    Args:
        preprocessed_path: Path to the preprocessed output file.

    Returns:
        A boolean value indicating if the volume is already preprocessed.
    """
    preprocessed_path = Path(preprocessed_path)
    if preprocessed_path.exists() and preprocessed_path.stat().st_size > 0:
        return True

    return False


def main():

    # get arguments: first and last index to segment and the job id
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    job_id = sys.argv[3]
    config_key = sys.argv[4]
    config = CONFIGS[config_key]

    # Paths
    LABELS_PATH = PROJECT_ROOT / "data/processed/labels_cleaned_with_split.csv"
    PREPROCESSED_ROOT = Path("/anvme/workspace/iwi5437h-ct-anomaly-detection/preprocessed") / config["name"]
    LOG_DIR = Path("/anvme/workspace/iwi5437h-ct-anomaly-detection/logs/preprocessing") / config["name"]

    job_start_time = time.time()

    print(f"[Job {job_id}] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Job {job_id}] Processing volumes {start_index} to {end_index}")

    # define log files for successful segmentaions and failed ones
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    preprocessed_log = LOG_DIR / f"preprocessed_job{job_id}.txt"
    not_preprocessed_log = LOG_DIR / f"not_preprocessed_job{job_id}.txt"

    
    volume_list_df = pd.read_csv(LABELS_PATH)
    preprocessing_batch = volume_list_df.iloc[start_index:end_index]

    preprocessed_count = 0
    not_preprocessed_count = 0
    preprocessed_before_count = 0

    # start preprocessing each volume in the batch
    for i, row in enumerate(preprocessing_batch.itertuples()):

        # continue preprocessing if we have at least more than 10 minutes to the maximum time 
        running_time = time.time() - job_start_time
        remaining_time = MAX_JOB_SECONDS - running_time

        if remaining_time < 600:  
            print(f"[STOP Job {job_id}] Only {remaining_time/60:.1f} minutes remaining")
            print(f"[Job {job_id}] Processed {i} volumes in this run, {len(preprocessing_batch) - i} volumes are remaining")
            break
    
        volume_name = row.volume_name.replace(".nii.gz", "")
        
        preprocessed_path = PREPROCESSED_ROOT / f"{volume_name}.npz"

        # skip if this volume already segmented before
        if is_preprocessed(preprocessed_path):
            preprocessed_before_count += 1
            print(f"Job {job_id} {i+1}-th volume in this batch already preprocessed before.")
            continue

        print(f"Job {job_id}: {i+1}-th volume processing: {volume_name}")

        try:
            preprocess_one_volume(
                volume_path=row.volume_data_path,
                masks_dir=row.mask_dir,
                preprocessed_path=preprocessed_path,
                target_voxel_spacing=config["target_voxel_spacing"],
                target_shape=config["target_shape"],
                hu_min=config["hu_min"],
                hu_max=config["hu_max"],
                bbox_margin=config["bbox_margin"],
                lung_only=config["lung_only"],
            )
            
            with open(preprocessed_log, "a") as f:
                f.write(f"{volume_name}\n")

            preprocessed_count += 1
            print(f"Job {job_id}: {i+1}-th volume Done: {volume_name}")
            
        except Exception:
            not_preprocessed_count += 1
            with open(not_preprocessed_log, "a") as f:
                f.write(f"{volume_name}:\n{traceback.format_exc()}\n")
            print(f"Job {job_id}: {i+1}-th volume FAILED: {volume_name}")


    print()
    print(f"[Job {job_id}] Finished preprocessing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Job {job_id}] Preprocessed: {preprocessed_count} | Skipped (Already Preprocessed): {preprocessed_before_count} | Failed: {not_preprocessed_count}")


if __name__ == "__main__":
    main()