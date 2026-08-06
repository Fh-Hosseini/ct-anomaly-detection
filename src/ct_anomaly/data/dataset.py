"""
Dataset class
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CTDataset(Dataset):

    def __init__(self, labels_path, preprocessed_dir, split):

        labels_df = pd.read_csv(labels_path)
        
        
        self.labels_df = labels_df[labels_df["split"] == split].reset_index(drop=True)
        if len(self.labels_df) == 0:
            raise ValueError(f"Nothing is in {split} split.")

        self.preprocessed_dir = Path(preprocessed_dir)
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):

        # read and load the volume data
        row = self.labels_df.iloc[idx]
        
        npz_filename = row["volume_name"].replace(".nii.gz", ".npz")
        volume_path = self.preprocessed_dir / npz_filename
        npz_file = np.load(volume_path)
        volume = npz_file["volume"]

        # convert to torch tensor
        volume_tensor = torch.from_numpy(volume).float()

        # add channel dim, as Conv3d requires (C, D, H, W)
        volume_tensor = volume_tensor.unsqueeze(0)

        label = int(row["binary_label"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        return volume_tensor, label_tensor, row["volume_name"]