"""
This module provides functions for resizing 3D volumes using interpolation methods.
It includes a function to downsample a 3D volume to a specified target shape.
"""

import torch


def downsample_volume(volume, target_shape, mode="trilinear"):
    """
    Downsample a 3D volume to a target shape using interpolation.

    Args:
        volume: The input 3D volume tensor of shape (C, D, H, W).
        target_shape: The desired output shape.
        mode: The interpolation mode to use. Default is 'trilinear'.

    Returns:
        torch.Tensor: The downsampled volume tensor of shape.
    """
    if tuple(volume.shape[2:]) == tuple(target_shape):
        return volume

    return torch.nn.functional.interpolate(
        volume,
        size=target_shape,
        mode=mode,
        align_corners=False,
    )
