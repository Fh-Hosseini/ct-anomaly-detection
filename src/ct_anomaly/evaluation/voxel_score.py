"""
This module provides a function to compute anomaly scores based on the reconstruction error between original volumes and their
reconstructions.
The anomaly scores can be computed using different aggregation strategies, including mean, max, and top-k
"""

import torch
from monai.networks.layers import GaussianFilter


def compute_voxel_scores(volumes, reconstructions, aggregation, top_k_fraction, smoothing_sigma):
    """
    Compute anomaly scores based on the reconstruction error between volumes and reconstructions.

    Args:
        volumes: The original volumes, shape (B, C, D, H, W).
        reconstructions: The reconstructed volumes, shape (B, C, D, H, W).
        aggregation: The aggregation strategy to compute the final score. One of "mean", "max", or "topk".
        top_k_fraction: The fraction of top-k errors to consider when using the "topk" aggregation strategy.
        smoothing_sigma: The standard deviation for Gaussian smoothing. If 0, no smoothing is applied.

    Returns:
        torch.Tensor: The computed anomaly scores, shape (B,).
    """

    if volumes.shape != reconstructions.shape:
        raise ValueError(
            "volumes and reconstructions must have the same shape, got "
            + str(tuple(volumes.shape)) + " and " + str(tuple(reconstructions.shape))
        )

    error_map = (volumes.float() - reconstructions.float()) ** 2

    if smoothing_sigma > 0:
        smoothing = GaussianFilter(spatial_dims=3, sigma=smoothing_sigma)
        smoothing = smoothing.to(error_map.device)
        error_map = smoothing(error_map)

    batch_size = error_map.shape[0]
    flat_errors = error_map.reshape(batch_size, -1)

    if aggregation == "mean":
        scores = flat_errors.mean(dim=1)

    elif aggregation == "max":
        scores = flat_errors.max(dim=1).values

    elif aggregation == "topk":
        num_voxels = flat_errors.shape[1]
        num_top = int(round(num_voxels * top_k_fraction))
        if num_top < 1:
            num_top = 1
        top_values = torch.topk(flat_errors, k=num_top, dim=1).values
        scores = top_values.mean(dim=1)

    else:
        raise ValueError(
            "Unknown aggregation strategy: " + str(aggregation)
            + ". Expected one of: mean, max, topk."
        )

    return scores
