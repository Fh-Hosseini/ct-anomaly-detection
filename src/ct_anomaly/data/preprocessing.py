"""
This module provides functions for preprocessing CT volumes, including loading, cropping to lung bounding boxes, 
resampling, resizing, clipping HU values, normalizing, and saving the preprocessed volumes.
The preprocessing steps are designed to prepare the CT volumes for further analysis or model training.

Preprocessing steps:
1. Load the CT volume from a NIfTI file.
2. Crop the volume to the bounding box of the lungs using the provided lung masks.
3. Resample the volume to a target voxel spacing.
4. Resize the volume to a fixed target shape.
5. Clip the HU values to a specified range and normalize them to the range [-1, +1].
6. Save the preprocessed volume as a compressed .npz file.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
from src.ct_anomaly.data.segmentation import combine_lung_lobes_masks, get_bounding_box


# Target voxel spacing in mm --- to try: (1.5, 1.5, 1.5) or (0.75, 0.75, 1.5)
TARGET_VOXEL_SPACING = (0.75, 0.75, 1.5)

# Target shape after resampling --- to try: (128, 128, 128) or (480, 480, 240)
TARGET_SHAPE = (480, 480, 240)

# HU clipping range 
TARGET_HU_MIN = -1000
TARGET_HU_MAX = 400 # to try: 200, 400, 1000

# Margin to add around the bounding box when cropping
BBOX_MARGIN = 10


def load_volume(volume_path):
    """
    Load a CT volume from a NIfTI file.

    Args:
        volume_path: Path to the input CT volume in NIfTI format.
    
    Returns:
        volume: 3D numpy array representing the loaded volume.
        current_voxel_spacing: Current voxel spacing in mm (x, y, z).
    """
    nii = nib.load(volume_path)
    volume = nii.get_fdata().astype(np.float32)
    current_voxel_spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    return volume, current_voxel_spacing


def crop_to_lung_bounding_box(volume, masks_dir, margin=0):
    """
    Crop a 3D volume to the bounding box of the lungs using the provided lung masks.

    Args:
        volume: 3D numpy array representing the volume to be cropped.
        masks_dir: Directory containing the lung masks for the volume.
        margin: Margin to add around the bounding box (default: 0).

    Returns:
        Cropped volume as a 3D numpy array.
    """

    masks_dir = Path(masks_dir)
    
    lung_mask = combine_lung_lobes_masks(masks_dir)
    bbox = get_bounding_box(lung_mask)

    # Crop the volume to the bounding box, add 1 because of python slicing 
    cropped_lung_volume = volume[
        max(0, bbox["x_min"] - margin) : min(bbox["x_max"] + margin, volume.shape[0]) + 1,
        max(0, bbox["y_min"] - margin) : min(bbox["y_max"] + margin, volume.shape[1]) + 1,
        max(0, bbox["z_min"] - margin) : min(bbox["z_max"] + margin, volume.shape[2]) + 1
    ]
    return cropped_lung_volume


def resample_volume(volume, current_voxel_spacing, target_voxel_spacing=TARGET_VOXEL_SPACING):
    """
    Resample a 3D volume to the target voxel spacing using trilinear interpolation.

    Args:
        volume: 3D numpy array representing the volume to be resampled.
        current_voxel_spacing: Current voxel spacing in mm (x, y, z).
        target_voxel_spacing: Target voxel spacing in mm (x, y, z).

    Returns:
        Resampled volume as a 3D numpy array with the target voxel spacing.
    """

    # Calculate the zoom factors for each dimension based on the current and target spacing
    zoom_factors = tuple(current_voxel_spacing[i] / target_voxel_spacing[i] for i in range(3))

    # Resample the volume using trilinear interpolation
    resampled_volume = zoom(volume, zoom_factors, order=1).astype(np.float32)

    return resampled_volume


def resize_volume(volume, target_shape=TARGET_SHAPE):
    """
    Resize a 3D volume to the target shape using trilinear interpolation.

    Args:
        volume: 3D numpy array representing the volume to be resized.
        target_shape: Target shape (x, y, z) for the output volume.

    Returns:
        Resized volume as a 3D numpy array with the target shape.
    """

    # Calculate the zoom factors for each dimension based on the current and target shape
    zoom_factors = tuple(target_shape[i] / volume.shape[i] for i in range(3))

    # Resize the volume using trilinear interpolation
    resized_volume = zoom(volume, zoom_factors, order=1).astype(np.float32)

    return resized_volume


def clip_and_normalize(volume, hu_min=TARGET_HU_MIN, hu_max=TARGET_HU_MAX):
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



def preprocess_volume(volume_path, masks_dir, output_path):
    """
    Preprocess a CT volume by loading, cropping to lung bounding box, resampling, resizing, clipping HU values, normalizing, and saving the preprocessed volume.

    Args:
        volume_path: Path to the input CT volume in NIfTI format.
        masks_dir: Directory containing the lung masks for the volume.
        output_path: Path to save the preprocessed volume as a .npz file.
    """
    
    print(f"Processing: {Path(volume_path).name}")

    # Load
    volume, spacing = load_volume(volume_path)

    # Crop to lung bounding box
    volume = crop_to_lung_bounding_box(volume, masks_dir)
    print(f"    After crop: {volume.shape}")

    # Resample to target spacing
    volume = resample_volume(volume, spacing)
    print(f"    After resample: {volume.shape}")

    # Resize to fixed size
    volume = resize_volume(volume)
    print(f"    After resize: {volume.shape}")

    # Clip and normalize
    volume = clip_and_normalize(volume)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, volume=volume)
    print(f"    Saved to: {output_path}")
    print(f"    Done: output shape:{volume.shape}")