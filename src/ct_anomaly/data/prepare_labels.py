"""
This script is used to prepare and clean the labels df for the CT anomaly detection project.

 It performs the following tasks:

1. Load the raw label Excel file and parse the VolumeName into patient_id, scan_id, and reconstruction.
2. Exclude brain scans from the dataframe based on the provided lists of brain scan file names.
3. Exclude invalid volumes from the dataframe based on the provided list of invalid volume names.
4. Create binary labels: 0 for healthy scans and 1 for unhealthy scans by combining borderline and unhealthy labels.
5. Add two new columns to the dataframe: "volume_data_path" and "mask_dir" which contain the paths to the CT volume data and the lung mask directory.
6. Save the cleaned and processed dataframe to a CSV file.      

"""

import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
CT_DATA_ROOT = Path("/anvme/workspace/b180dc29-CT_RATE_IDEA_MIRROR")
MASKS_ROOT = Path("/anvme/workspace/iwi5437h-ct-anomaly-detection/lung_masks")

BRAIN_TRAIN_PATH = PROJECT_ROOT / "data/raw/no_chest_train.txt"
BRAIN_VALID_PATH = PROJECT_ROOT / "data/raw/no_chest_valid.txt"
INVALID_SEGMENTATION_LIST_PATH = PROJECT_ROOT / "data/processed/seg_invalid_volumes.csv"
INVALID_PREPROCESSING_LIST_PATH = PROJECT_ROOT / "data/processed/preprocess_invalid_volumes.csv"

LABELS_DF_PATH = PROJECT_ROOT / "data/raw/CT-RATE_reports_full_gpt-oss-120b.xlsx"
PROCESSED_LABELS_PATH = PROJECT_ROOT / "data/processed/labels_cleaned.csv"


def parse_volume_name(volume_name):
    """
    Parses the volume name to extract CT-image data such as patient ID, scan ID, and reconstruction number.

    Args:
        volume_name: The name of the volume, expected in the format "PatientID_ScanID_Reconstruction.nii.gz"

    Returns:
        dict: A dictionary containing the extracted data with keys "patient_id", "scan_id", and "reconstruction".
    """

    volume_metadata = {"patient_id": None, "scan_id": None, "reconstruction": None}

    str_parts = volume_name.replace(".nii.gz", "").split("_")

    if len(str_parts) == 4:
        volume_metadata["patient_id"] = str_parts[0] + "_" + str_parts[1]
        volume_metadata["scan_id"] = str_parts[2]
        volume_metadata["reconstruction"] = int(str_parts[3])

    return volume_metadata


def load_and_parse_labels_df(filepath):
    """
    Load raw label Excel file, drop empty columns, parse VolumeName into 
    patient_id, scan_id, and reconstruction.

    Args:
        filepath: The path to the Excel file containing the labels.

    Returns:
        df: A pandas DataFrame containing the parsed metadata.
    """

    df = pd.read_excel(filepath)

    # drop unnamed and empty columns
    df = df[["Predicted_label", "VolumeName", "Findings_EN", "Impressions_EN"]].copy()

    # Rename columns to snake_case
    df = df.rename(columns={
    "Predicted_label": "predicted_label",
    "VolumeName": "volume_name",
    "Findings_EN": "findings_en",
    "Impressions_EN": "impressions_en",
})

    # parse volume_name: patient_id, scan_id, reconstruction
    df_volume_data = df["volume_name"].apply(parse_volume_name).apply(pd.Series)
    df_parsed = pd.concat([df, df_volume_data], axis=1)

    return df_parsed


def _read_brain_scans_names(filepath):
    """
    Read the file names from a text file and extract the name of the volumes which can be used to find the brain scans.

    Args:
        filepath: The path to the text file containing the file paths.

    Returns:
        brain_scans: A set of brain scan file names extracted from the file paths.
    """

    with open(filepath) as f:
        file_paths = f.read().splitlines()

    brain_scans = set()
    for file_path in file_paths:
        file_name = file_path.split("/")[-1]
        brain_scans.add(file_name)

    return brain_scans


def exclude_brain_scans(df, brain_train_path, brain_valid_path):
    """
    Exclude brain scans from the dataframe based on the provided lists of brain scan file names.

    Args:
        df: The dataframe containing information about all volumes and their labels.
        brain_train_path: The path to the text file containing the file paths for the CT_RATE training set.
        brain_valid_path: The path to the text file containing the file paths for the CT_RATE validation set.

    Returns:
        df_filtered: A filtered dataframe that excludes the brain scans.
    """

    brain_scans_train = _read_brain_scans_names(brain_train_path)
    brain_scans_valid = _read_brain_scans_names(brain_valid_path)
    brain_scans = brain_scans_train | brain_scans_valid

    print(f"Number of brain scans to exclude: {len(brain_scans)}")

    brain_mask = df["volume_name"].isin(brain_scans)
    df_filtered = df[~brain_mask].copy()

    print(f"\nNumber of all volumes before filtering: {len(df)}")
    print(f"Number of volumes after filtering the brain scans: {len(df_filtered)}")

    return df_filtered


def exclude_invalid_volumes(df, invalid_list_path):
    """
    Exclude invalid volumes from the dataframe.

    Args:
        df: The dataframe containing information about all volumes and their labels.
        invalid_list_path: The path to the CSV file containing the list of invalid volume names. 
    
    Returns:
        df_filtered: A filtered dataframe that excludes the invalid volumes.
    """

    invalid_vol_df = pd.read_csv(invalid_list_path)
    invalid_volumes = set(invalid_vol_df["volume_name"].tolist())

    invalid_mask = df["volume_name"].str.replace(".nii.gz", "").isin(invalid_volumes)
    df_filtered = df[~invalid_mask].copy()

    print(f"Removed {len(invalid_volumes)} invalid volumes")

    return df_filtered


def create_binary_labels(df):
    """
    Create binary labels: 0 for healthy scans and 1 for unhealthy scans by combining borderline and unhealthy labels.
    As all the reconstructions of a scan should have the same label, we assign the maximum label for all reconstructions of the same scan.

    Args:
        df: The metadata dataframe containing information about all volumes, including the original predicted labels. 

    Returns:
        scan_df: A dataframe at the scan level containing the binary labels and the number of volumes per scan. 
        df: The original dataframe with an additional column for binary labels at the volume level.
    """

    # combine unhealthy(2) and borderline(1) labels into one class (unhealthy) and keep healthy(0) as is
    df = df.copy()
    df["binary_label"] = df["predicted_label"].apply(lambda x: 0 if x == 0 else 1)

    # TODO MAYBE UNCOMMENT THIS
    # assign the maximum label for all reconstructions of the same scan
    # scan_max_label = df.groupby(["patient_id", "scan_id"])["binary_label"].transform("max")
    # df["binary_label"] = scan_max_label

    return df


def get_volume_data_path(volume_name, patient_id, scan_id):
    """
    Find volume path using the volume information: volume_name, patient_id, scan_id.

    Args: 
        volume_name: name of the volume
        patient_id: id of the patient
        scan_id: id of the scan

    Return:
        data_path: path to the volume CT data
    """

    # the split here is either "train", "val" and it is from the data split of CT Rate dataset itself, not the split used for training in our project.
    split = patient_id.split("_")[0] 
    data_path = CT_DATA_ROOT / f"CT-RATE_{split}_fixed" / patient_id / f"{patient_id}_{scan_id}" / f"{volume_name}.nii.gz"

    return data_path


def add_path_columns(df):
    """
    Add two new columns to the dataframe: "volume_data_path" and "mask_dir".

    Args:
        df: The dataframe containing the volume information.
    Returns:
        df: The dataframe with the new columns added.
    """

    df = df.copy()

    # add a column for the ct volume data path
    df["volume_data_path"] = df.apply(
        lambda row: str(get_volume_data_path(
            row["volume_name"].replace(".nii.gz", ""),
            row["patient_id"],
            row["scan_id"]
        )), axis=1
    )

    # add a column for the lung mask directory path
    df["mask_dir"] = df["volume_name"].apply(
        lambda v: str(MASKS_ROOT / v.replace(".nii.gz", ""))
    )

    return df


def main():

    df = load_and_parse_labels_df(LABELS_DF_PATH)
    print(f"Loaded labels with {len(df)} number of rows")

    df_filtered = exclude_brain_scans(df, BRAIN_TRAIN_PATH, BRAIN_VALID_PATH)
    df_filtered = exclude_invalid_volumes(df_filtered, INVALID_SEGMENTATION_LIST_PATH)
    df_filtered = exclude_invalid_volumes(df_filtered, INVALID_PREPROCESSING_LIST_PATH)

    df_binarized = create_binary_labels(df_filtered)

    df_final = add_path_columns(df_binarized)

    print("Total number of volumes after processing:", len(df_final))
    print("\nFirst rows of processed df: ")
    print(df_final.head())


    PROCESSED_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PROCESSED_LABELS_PATH, index=False)
    print(f"\nSaved cleaned labels df to {PROCESSED_LABELS_PATH}")


if __name__ == "__main__":
    main()




