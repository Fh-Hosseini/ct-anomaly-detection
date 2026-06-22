"""
Create a data split for the CT-RATE dataset, ensuring that all volumes from the same scan of a patient are in the same split.

"""

import os

import pandas as pd

from sklearn.model_selection import train_test_split


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


def load_and_parse_labels(filepath):
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

    # parse VolumeName: patient_id, scan_id, reconstruction
    df_volume_data = df["VolumeName"].apply(parse_volume_name).apply(pd.Series)
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

    brain_mask = df["VolumeName"].isin(brain_scans)
    df_filtered = df[~brain_mask].copy()

    print(f"\nNumber of all volumes before filtering: {len(df)}")
    print(f"Number of volumes after filtering the brain scans: {len(df_filtered)}")

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
    df["binary_label"] = df["Predicted_label"].apply(lambda x: 0 if x == 0 else 1)

    # assign a label to each scan based on the maximum label of its volumes (if any volume is unhealthy, the whole scan is considered unhealthy)
    scan_label = df.groupby(["patient_id", "scan_id"])["binary_label"].max()

    # number of volumes per scan
    n_volumes_per_scan = df.groupby(["patient_id", "scan_id"])["VolumeName"].count()

    scan_df = pd.DataFrame({
        "scan_label": scan_label,
        "n_volumes": n_volumes_per_scan,
    }).reset_index()

    return scan_df, df


def _split_train_val_test(df, random_state = 42):
    """
    Split the df into three sets: train(70%), val(15%), and test(15%)

    Args:
        df: The dataframe to be split.
        random_state: The random state for reproducibility. 
    
    Returns:
        train: The training set dataframe.
        val: The validation set dataframe.
        test: The test set dataframe.
    """

    train, tmp = train_test_split(df, test_size=0.30, random_state=random_state) # split into 70% train and 30% temp (val + test)
    val, test = train_test_split(tmp, test_size=0.50, random_state=random_state) # split temp into 50% val and 50% test (15% each of the original data)
    return train, val, test



def create_scan_split(scan_df, df_filtered, random_state=42):
    """
    Create a data split, ensuring that all volumes from the same scan of a patient are in the same split. 
    The split is done separately for healthy and unhealthy scans to maintain class balance across the splits.

    Args:
        scan_df: Scan level dataframe with binary labels
        df_filtered: The filtered dataframe containing information about all volumes.
        random_state: The random state for reproducibility. 

    Returns:
        df_final: The final dataframe with the split information merged back to the volume level.
    """

    # separate healthy and unhealthy scans
    healthy_scans = scan_df[scan_df["scan_label"] == 0]
    anomalous_scans = scan_df[scan_df["scan_label"] == 1]

    # split healthy and unhealthy scans separately to maintain class balance across the splits
    healthy_train, healthy_val, healthy_test = _split_train_val_test(healthy_scans, random_state)
    anomalous_train, anomalous_val, anomalous_test = _split_train_val_test(anomalous_scans, random_state)

    # add the split column to each df
    healthy_train["split"] = "train"
    healthy_val["split"] = "val"
    healthy_test["split"] = "test"

    anomalous_train["split"] = "train"
    anomalous_val["split"] = "val"
    anomalous_test["split"] = "test"

    # concatenate all the dataframes to get the final scan level dataframe with the split information
    scan_df_split = pd.concat([
        healthy_train, healthy_val, healthy_test,
        anomalous_train, anomalous_val, anomalous_test,
    ])

    print(f"Total number of Scans: {len(scan_df_split)}")
    print("\nData split distribution:")
    print(scan_df_split["split"].value_counts())


    # map split back onto the volume level dataframe
    df_final = df_filtered.merge(
        scan_df_split[["patient_id", "scan_id", "split"]],
        on=["patient_id", "scan_id"],
        how="left",
    )

    print(f"\nTotal volumes: {len(df_final)}")
    print(df_final["split"].value_counts())
    print("\nVolumes per split and label:")
    print(df_final.groupby(["split", "binary_label"])["VolumeName"].count())

    return df_final


def main():

    labels_file = "data/raw/CT-RATE_reports_full_gpt-oss-120b.xlsx"
    brain_train_path = "data/raw/no_chest_train.txt"
    brain_valid_path = "data/raw/no_chest_valid.txt"
    output_file = "data/processed/labels_with_split.csv"

    df = load_and_parse_labels(labels_file)
    df_filtered = exclude_brain_scans(df, brain_train_path, brain_valid_path)
    
    scan_df_binarized_labels, df_binarized = create_binary_labels(df_filtered)
    df_final = create_scan_split(scan_df_binarized_labels, df_binarized)

    os.makedirs("data/processed", exist_ok=True)
    
    df_final.to_csv(output_file, index=False)
    print(f"\nSaved split to {output_file}")



if __name__ == "__main__":
    main()