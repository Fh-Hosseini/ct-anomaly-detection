"""
Preprocessing pipeline for CT-RATE-AD.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
import sys
from src.ct_anomaly.data.segmentation import combine_lung_masks, get_bounding_box


# Target voxel spacing in mm
VOXEL_SPACING = (0.75, 0.75, 1.5)

# Target shape after resampling
TARGET_SHAPE = (480, 480, 240)

# HU clipping range 
HU_MIN = -1000
HU_MAX = 1000


def resample_volume(volume, current_spacing, target_spacing=VOXEL_SPACING,):
    """
    Resample a 3D volume to the target voxel spacing using trilinear interpolation.

    Args:
        volume: 3D numpy array representing the volume to be resampled.
        current_spacing: Current voxel spacing in mm (x, y, z).
        target_spacing: Target voxel spacing in mm (x, y, z).

    Returns:
        Resampled volume as a 3D numpy array with the target voxel spacing.
    """

    # Calculate the zoom factors for each dimension based on the current and target spacing
    zoom_factors = tuple(current_spacing[i] / target_spacing[i] for i in range(3))

    # Resample the volume using trilinear interpolation
    resampled = zoom(volume, zoom_factors, order=1)

    return resampled.astype(np.float32)


def center_crop_or_pad(volume, target_shape=TARGET_SHAPE):
    """
    Center-crop or pad a 3D volume to the target shape.

    Args:
        volume: 3D numpy array representing the volume to be cropped or padded.
        target_shape: Target shape (x, y, z) for the output volume.

    Returns:
        Cropped or padded volume as a 3D numpy array with the target shape.
    """

    result = np.full(target_shape, fill_value=HU_MIN, dtype=np.float32)

    # Iterate over each dimension (x, y, z) to crop or pad
    for dim in range(3):
        current = volume.shape[dim]
        target = target_shape[dim]

        if current >= target:
            # Crop: find the start index for cropping
            start = (current - target) // 2
            volume = np.take(volume, range(start, start + target), axis=dim)
        else:
            # Pad: calculate the amount of padding needed before and after
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (pad_before, pad_after)
            volume = np.pad(volume, pad_width, mode="constant", constant_values=HU_MIN)

    result = volume.astype(np.float32)
    return result



def clip_and_normalize(volume, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    Clip the HU values of a 3D volume to a specified range and normalize them to [-1, +1].

    Args:
        volume: 3D numpy array representing the volume to be clipped and normalized.
        hu_min: Minimum HU value for clipping (default: -1000).
        hu_max: Maximum HU value for clipping (default: +1000).

    Returns:
        Clipped and normalized volume as a 3D numpy array with values in the range [-1, +1].
    """

    # Clip to HU range
    volume = np.clip(volume, hu_min, hu_max)

    # Normalize to [-1, +1]
    volume = (volume - hu_min) / (hu_max - hu_min)
    volume = volume * 2 - 1   

    return volume.astype(np.float32)


def preprocess_volume(volume_path, mask_dir, output_path):
    """
    Preprocess a CT volume by cropping to the lung bounding box, resampling, center-cropping or padding, clipping HU values, normalizing, and saving as a .npz file.

    Args:
        volume_path: Path to the input CT volume in NIfTI format.
        mask_dir: Directory containing the lung masks for the volume.
        output_path: Path to save the preprocessed volume as a .npz file.
    """
    
    sys.path.insert(0, str(Path(__file__).parents[3]))
    

    # Load the volume and get its current voxel spacing
    nii = nib.load(volume_path)
    volume = nii.get_fdata().astype(np.float32)
    current_spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])

    # Crop to lung bounding box
    lung_mask = combine_lung_masks(mask_dir)
    bbox = get_bounding_box(lung_mask)

    # Crop the volume to the bounding box, add 1 because of python slicing 
    volume = volume[
        bbox["x_min"]:bbox["x_max"] + 1,
        bbox["y_min"]:bbox["y_max"] + 1,
        bbox["z_min"]:bbox["z_max"] + 1,
    ]

    # Resample the volume to the target voxel spacing
    volume = resample_volume(volume, current_spacing)

    # Center-crop or pad the volume to the target shape
    volume = center_crop_or_pad(volume)

    # Clip and normalize the volume to the specified HU range and normalize to [-1, +1]
    volume = clip_and_normalize(volume)

    # Save the preprocessed volume as a compressed .npz file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, volume=volume)