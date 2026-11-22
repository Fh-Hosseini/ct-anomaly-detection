
from monai.networks.layers import GaussianFilter
from src.ct_anomaly.evaluation.aggregate import aggregate_map


def compute_voxel_scores(volumes, reconstructions, aggregation, top_k_fraction, smoothing_sigma):

    if volumes.shape != reconstructions.shape:
        raise ValueError(
            "volumes and reconstructions must have the same shape, got "
            + str(tuple(volumes.shape))
            + " and "
            + str(tuple(reconstructions.shape))
        )

    error_map = (volumes.float() - reconstructions.float()) ** 2

    if smoothing_sigma > 0:
        smoothing = GaussianFilter(spatial_dims=3, sigma=smoothing_sigma)
        smoothing = smoothing.to(error_map.device)
        error_map = smoothing(error_map)

    return aggregate_map(error_map, aggregation, top_k_fraction)
