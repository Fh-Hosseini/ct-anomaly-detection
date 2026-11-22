"""
Dataset class
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CTDataset(Dataset):
    def __init__(self, labels_path, preprocessed_dir, split, healthy_only=False):
        labels_df = pd.read_csv(labels_path)
        split_df = labels_df[labels_df["split"] == split]

        if healthy_only:
            split_df = split_df[split_df["binary_label"] == 0]

        self.labels_df = split_df.reset_index(drop=True)

        if len(self.labels_df) == 0:
            raise ValueError(f"Nothing is in {split} split.")

        self.preprocessed_dir = Path(preprocessed_dir)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        npz_filename = row["volume_name"].replace(".nii.gz", ".npz")
        volume_path = self.preprocessed_dir / npz_filename
        with np.load(volume_path) as npz_file:
            volume = npz_file["volume"]

        volume_tensor = torch.from_numpy(volume).float()
        volume_tensor = volume_tensor.unsqueeze(0)

        label = int(row["binary_label"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        return volume_tensor, label_tensor, row["volume_name"]
