"""
Create a data split for the CT-RATE dataset, ensuring that all volumes from the same scan of a patient are in the same split.

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
CLEANED_LABELS_PATH = PROJECT_ROOT / "data/processed/labels_cleaned.csv"
LABELS_WITH_SPLIT_PATH = PROJECT_ROOT / "data/processed/labels_cleaned_with_split.csv"



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



def create_scan_split(df, random_state=42):
    """
    Create a data split, ensuring that all volumes from the same scan of a patient are in the same split. 
    The split is done separately for healthy and unhealthy scans to maintain class balance across the splits.

    Args:
        df: The metadata dataframe containing information about all volumes, including the original predicted labels.
        random_state: The random state for reproducibility. 

    Returns:
        df_final: The final dataframe with the split information merged back to the volume level.
    """

    scan_df = df.groupby(["patient_id", "scan_id"])["binary_label"].max().reset_index()
    scan_df = scan_df.rename(columns={"binary_label": "scan_label"})

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
    df_split = df.merge(
        scan_df_split[["patient_id", "scan_id", "split"]],
        on=["patient_id", "scan_id"],
        how="left",
    )

    print(f"\nTotal volumes: {len(df_split)}")
    print(df_split["split"].value_counts())
    print("\nVolumes per split and label:")
    print(df_split.groupby(["split", "binary_label"])["volume_name"].count())

    return df_split


def main():

    df = pd.read_csv(CLEANED_LABELS_PATH)
    df_final = create_scan_split(df, random_state=42)

    LABELS_WITH_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(LABELS_WITH_SPLIT_PATH, index=False)
    print(f"\nSaved df with data split to {LABELS_WITH_SPLIT_PATH}")


if __name__ == "__main__":
    main()