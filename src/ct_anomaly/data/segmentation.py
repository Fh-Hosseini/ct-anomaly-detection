"""
Segment five lobes of lung using TotalSegmentator.

Includes Functions to:
- Segment the 5 lung lobes from a CT volume using TotalSegmentator
- Combine lobe masks into a single binary lung mask
- Compute the bounding box of the lung area for cropping from the images
"""

import os
import tempfile
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


# The 5 lung lobes of TotalSegmentator
LUNG_LOBES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


def segment_one_volume(volume_path, mask_dir, device="gpu"):
    """
    Run TotalSegmentator on a single CT volume to extract lung lobe masks.

    Args:
        volume_path: Path to the input .nii.gz CT volume file
        mask_dir: Directory to save the output lung lobe masks
        device: Device to run on
    """

    tempfile.tempdir = os.environ.get("TMPDIR", "/tmp")
    mask_dir.mkdir(parents=True, exist_ok=True)

    nii = nib.load(volume_path)

    totalsegmentator(
        input=nii,
        output=mask_dir,
        roi_subset=LUNG_LOBES,
        device=device,
        quiet=True,
    )


def combine_lung_lobes_masks(mask_dir):
    """
    Combine the individual lung lobes masks into a single binary lung mask.

    Args:
        mask_dir: Directory containing the individual lung lobes masks from TotalSegmentator

    Returns:
        combined_mask: Binary numpy array where 1 indicates lung and 0 indicates background
    """

    combined_mask = None

    for lobe in LUNG_LOBES:
        mask_path = mask_dir / f"{lobe}.nii.gz"
        lobe_mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        if combined_mask is None:
            combined_mask = lobe_mask
        else:
            combined_mask = np.logical_or(combined_mask, lobe_mask).astype(np.uint8)

    return combined_mask


def get_bounding_box(lung_mask):
    """
    Get the bounding box coordinates of the lung region from a binary lung mask.
    
    Args:
        lung_mask: Binary numpy array where 1 indicates lung and 0 indicates background

    Returns:
        bbox_indices: A dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        representing the bounding box coordinates of the lung region.
    """

    # Find the indices of the lung voxels
    lung_voxel_indices = np.argwhere(lung_mask > 0)

    bbox_indices = {
        "x_min": int(lung_voxel_indices[:, 0].min()),
        "x_max": int(lung_voxel_indices[:, 0].max()),
        "y_min": int(lung_voxel_indices[:, 1].min()),
        "y_max": int(lung_voxel_indices[:, 1].max()),
        "z_min": int(lung_voxel_indices[:, 2].min()),
        "z_max": int(lung_voxel_indices[:, 2].max()),
    }

    return bbox_indices