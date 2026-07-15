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


def crop_to_lung_bounding_box(data, bbox, bbox_margin):
    """
    Crop a 3D volume to the bounding box of the lungs using the provided lung masks.

    Args:
        data: 3D numpy array representing the volume to be cropped.
        bbox: Dictionary containing the bounding box coordinates (x_min, x_max, y_min, y_max, z_min, z_max).
        bbox_margin: Margin to add around the bounding box (default: BBOX_MARGIN).

    Returns:
        Cropped volume as a 3D numpy array.
    """

    # Crop to lung bounding box using the combined lung mask
    cropped = data[
        max(0, bbox["x_min"] - bbox_margin) : min(bbox["x_max"] + bbox_margin, data.shape[0]) + 1,
        max(0, bbox["y_min"] - bbox_margin) : min(bbox["y_max"] + bbox_margin, data.shape[1]) + 1,
        max(0, bbox["z_min"] - bbox_margin) : min(bbox["z_max"] + bbox_margin, data.shape[2]) + 1
    ]

    return cropped

def apply_mask(volume, mask, hu_min):
    """
    Apply a binary mask to a 3D volume, setting values outside the mask to a specified minimum HU value.

    Args:
        volume: 3D numpy array representing the volume to be masked.
        mask: 3D binary numpy array representing the mask (1 for lung regions, 0 for non-lung regions).
        hu_min: Minimum HU value to set for voxels outside the mask (default: TARGET_HU_MIN).
    
    Returns:
        Masked volume as a 3D numpy array, with values outside the mask set to hu_min.
    """
    volume = volume.copy()
    volume[~mask.astype(bool)] = hu_min  # Set values outside the mask to hu_min
    return volume

def resample_volume(volume, current_voxel_spacing, target_voxel_spacing):
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


def resize_volume(volume, target_shape):
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


def clip_and_normalize(volume, hu_min, hu_max):
    """
    Clip the HU values of a 3D volume to a specified range and normalize them to [-1, +1].

    Args:
        volume: 3D numpy array representing the volume to be clipped and normalized.
        hu_min: Minimum HU value for clipping.
        hu_max: Maximum HU value for clipping.
        
    Returns:
        Clipped and normalized volume as a 3D numpy array with values in the range [-1, +1].
    """
    
    # Clip to HU range
    volume = np.clip(volume, hu_min, hu_max)

    # Normalize to [-1, +1]
    volume = (volume - hu_min) / (hu_max - hu_min)
    volume = volume * 2 - 1   

    return volume.astype(np.float32)



def preprocess_one_volume(volume_path, masks_dir, preprocessed_path, target_voxel_spacing, target_shape, hu_min, hu_max, bbox_margin, lung_only):
    """
    Preprocess a CT volume by loading, cropping to lung bounding box, resampling, resizing, clipping HU values, normalizing, and saving the preprocessed volume.

    Args:
        volume_path: Path to the input CT volume in NIfTI format.
        masks_dir: Directory containing the lung masks for the volume.
        preprocessed_path: Path to save the preprocessed volume as a .npz file.
        target_voxel_spacing: Target voxel spacing in mm (x, y, z) for resampling.
        target_shape: Target shape (x, y, z) for resizing.
        hu_min: Minimum HU value for clipping.
        hu_max: Maximum HU value for clipping.
        bbox_margin: Margin to add around the bounding box when cropping.
        lung_only: If True, apply the lung mask to the volume after cropping.
    """
    
    print(f"Processing: {Path(volume_path).name}")

    # get lung bounding box from masks
    masks_dir = Path(masks_dir)
    lung_mask = combine_lung_lobes_masks(masks_dir)
    bbox = get_bounding_box(lung_mask)

    # Load
    volume, spacing = load_volume(volume_path)

    cropped_volume = crop_to_lung_bounding_box(volume, bbox, bbox_margin)
    # Crop to lung bounding box
    if lung_only:
        cropped_mask = crop_to_lung_bounding_box(lung_mask, bbox, bbox_margin)
        cropped_volume = apply_mask(cropped_volume, cropped_mask, hu_min=hu_min)        

    print(f"    After crop: {cropped_volume.shape}")

    # Resample to target spacing
    volume_resampled = resample_volume(cropped_volume, spacing, target_voxel_spacing=target_voxel_spacing)
    print(f"    After resample: {volume_resampled.shape}")

    # Resize to fixed size
    volume_resized = resize_volume(volume_resampled, target_shape=target_shape)
    print(f"    After resize: {volume_resized.shape}")

    # Clip and normalize
    volume_normalized = clip_and_normalize(volume_resized, hu_min=hu_min, hu_max=hu_max)

    # Save
    preprocessed_path = Path(preprocessed_path)
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preprocessed_path, volume=volume_normalized)
    print(f"    Saved to: {preprocessed_path}")
    print(f"    Done: preprocessed shape:{volume_normalized.shape}")