"""
Create a data split for the CT-RATE dataset, ensuring that all volumes from the same patient are in the same split.

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
CLEANED_LABELS_PATH = PROJECT_ROOT / "data/processed/labels_cleaned.csv"
LABELS_WITH_SPLIT_PATH = PROJECT_ROOT / "data/processed/labels_cleaned_with_split.csv"



def _split_train_val_test(df, stratify_column, random_state = 42):
    """
    Split the df into three sets: train(70%), val(15%), and test(15%)

    Args:
        df: The dataframe to be split.
        stratify_column: The column to use for stratification to maintain class balance across the splits.
        random_state: The random state for reproducibility. 
    
    Returns:
        train: The training set dataframe.
        val: The validation set dataframe.
        test: The test set dataframe.
    """

    train, tmp = train_test_split(df, test_size=0.30, stratify=df[stratify_column], random_state=random_state) # split into 70% train and 30% temp (val + test)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp[stratify_column], random_state=random_state) # split temp into 50% val and 50% test (15% each of the original data)
    return train, val, test





def create_patient_split(df, random_state=42):
    """
    Create a data split, ensuring that all volumes from the same patient are in the same split. 

    Args:
        df: The metadata dataframe containing information about all volumes, including the original predicted labels.
        random_state: The random state for reproducibility. 

    Returns:
        df_final: The final dataframe with the split information merged back to the volume level.
    """

    patient_df = df.groupby(["patient_id"])["binary_label"].max().reset_index()
    patient_df = patient_df.rename(columns={"binary_label": "patient_max_label"})

    train, val, test = _split_train_val_test(patient_df, stratify_column="patient_max_label", random_state=random_state)

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    # concatenate all the dataframes to get the final patient level dataframe with the split information
    patient_df_split = pd.concat([train, val, test])

    print(f"Total number of Patients: {len(patient_df_split)}")
    print("\nData split distribution:")
    print(patient_df_split["split"].value_counts())


    # map split back onto the volume level dataframe
    df_split = df.merge(
        patient_df_split[["patient_id", "split"]],
        on=["patient_id"],
        how="left",
    )

    print(f"\nTotal volumes: {len(df_split)}")
    print(df_split["split"].value_counts())
    print("\nVolumes per split and label:")
    print(df_split.groupby(["split", "binary_label"])["volume_name"].count())

    return df_split



def main():

    df = pd.read_csv(CLEANED_LABELS_PATH)
    df_final = create_patient_split(df, random_state=42)

    LABELS_WITH_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(LABELS_WITH_SPLIT_PATH, index=False)
    print(f"\nSaved df with data split to {LABELS_WITH_SPLIT_PATH}")


if __name__ == "__main__":
    main()