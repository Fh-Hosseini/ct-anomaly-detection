"""
Generate a CSV file that contains all CT volume names with their data paths and lung mask directories.
This used by the segmentation scripts later to know which volumes to process and segment.

Output: data/processed/volumes_path_list.csv
"""

import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
DATA_ROOT = Path("/anvme/workspace/b180dc29-CT_RATE_IDEA_MIRROR")
MASK_ROOT = Path("/anvme/workspace/iwi5437h-ct-anomaly-detection/lung_masks")
LABEL_PATH = PROJECT_ROOT / "data/processed/labels_with_split.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/volumes_path_list.csv"


def get_volume_data_path(volume_name, patient_id, scan_id):
    """
    Find volume path from the name of volume. 

    Args: 
        volume_name: name of the volume
        patient_id: id of the patient
        scan_id: id of the scan

    Return:
        data_path: path to the volume CT data
    """

    # the split here is either "train", "val" and it is from the data split of CT Rate dataset itself, not the split used for training in our project.
    split = patient_id.split("_")[0] 
    data_path = DATA_ROOT / f"CT-RATE_{split}_fixed" / patient_id / f"{patient_id}_{scan_id}" / f"{volume_name}.nii.gz"

    return data_path


def main():

    df = pd.read_csv(LABEL_PATH)
    print(f"Loaded labels with {len(df)} number of rows")

    rows = []

    # create a list contains path to each volume and their lung mask path
    for _, row in df.iterrows():
        volume_name = row["VolumeName"].replace(".nii.gz", "")
        patient_id = row["patient_id"]
        scan_id = row["scan_id"]

        volume_data_path = get_volume_data_path(volume_name, patient_id, scan_id)
        mask_dir = MASK_ROOT / volume_name

        rows.append({
            "volume_name": volume_name,
            "volume_data_path": volume_data_path,
            "mask_dir": mask_dir
        })

    # save the list 
    volumes_path_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    volumes_path_df.to_csv(OUTPUT_PATH, index=False)

    print("Total number of volume saved in the list:", len(volumes_path_df))
    print("Volume_path list saved.")
    print("\nFirst rows of this df: ")
    print(volumes_path_df.head())


if __name__ == "__main__":
    main()
