import torch
from monai.networks.layers import GaussianFilter
from src.ct_anomaly.evaluation.aggregate import aggregate_map


def compute_latent_precision_map(log_variance, target_shape):

    variance = torch.exp(log_variance)
    precision = 1.0 / variance
    precision = precision.mean(dim=1, keepdim=True)

    precision = torch.nn.functional.interpolate(
        precision, size=target_shape, mode="trilinear", align_corners=False
    )

    return precision


def compute_variance_weighted_scores(volumes, reconstructions, log_variance, aggregation, top_k_fraction, smoothing_sigma):
    volumes = volumes.float()
    reconstructions = reconstructions.float()
    log_variance = log_variance.float()

    squared_error = (volumes - reconstructions) ** 2

    spatial_shape = tuple(volumes.shape[2:])
    precision = compute_latent_precision_map(log_variance, spatial_shape)

    weighted_error = squared_error * precision

    if smoothing_sigma > 0:
        smoothing = GaussianFilter(spatial_dims=3, sigma=smoothing_sigma)
        smoothing = smoothing.to(weighted_error.device)
        weighted_error = smoothing(weighted_error)

    return aggregate_map(weighted_error, aggregation, top_k_fraction)


def compute_elbo_scores(volumes, reconstructions, mean, log_variance, beta):

    volumes = volumes.float()
    reconstructions = reconstructions.float()
    mean = mean.float()
    log_variance = log_variance.float()

    batch_size = volumes.shape[0]

    squared_error = (volumes - reconstructions) ** 2
    reconstruction_term = squared_error.reshape(batch_size, -1).mean(dim=1)

    num_input_voxels = volumes[0].numel()
    kl_per_element = -0.5 * (1.0 + log_variance - mean.pow(2) - log_variance.exp())
    kl_term = kl_per_element.reshape(batch_size, -1).sum(dim=1) / num_input_voxels
    return reconstruction_term + beta * kl_term


def compute_kl_only_scores(mean, log_variance):
    mean = mean.float()
    log_variance = log_variance.float()

    batch_size = mean.shape[0]

    kl_per_element = -0.5 * (1.0 + log_variance - mean.pow(2) - log_variance.exp())
    return kl_per_element.reshape(batch_size, -1).sum(dim=1)


def compute_kl_map(mean, log_variance):
    mean = mean.float()
    log_variance = log_variance.float()
    return -0.5 * (1.0 + log_variance - mean.pow(2) - log_variance.exp())


def compute_precision_map(log_variance):

    return torch.exp(-log_variance.float())


def compute_sampled_reconstructions(model, volumes, num_samples, generator=None):
    mean, log_variance = model.encoder(volumes)
    standard_deviation = torch.exp(0.5 * log_variance.float())

    reconstructions = []
    for _ in range(num_samples):
        noise = torch.randn(
            mean.shape, device=mean.device, dtype=torch.float32, generator=generator
        )
        sample = mean.float() + standard_deviation * noise
        reconstructions.append(model.decoder(sample))

    return torch.stack(reconstructions, dim=0)


def compute_sample_disagreement(sampled_reconstructions, aggregation, top_k_fraction, smoothing_sigma):

    variance_map = sampled_reconstructions.float().var(dim=0)

    if smoothing_sigma > 0:
        smoothing = GaussianFilter(spatial_dims=3, sigma=smoothing_sigma)
        smoothing = smoothing.to(variance_map.device)
        variance_map = smoothing(variance_map)

    return aggregate_map(variance_map, aggregation, top_k_fraction)
