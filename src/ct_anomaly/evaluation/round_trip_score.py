"""
Round-trip feature scores.
"""

import torch
import torch.nn.functional as F
from monai.networks.layers import GaussianFilter
from src.ct_anomaly.evaluation.aggregate import aggregate_map

EPSILON = 1e-8


def get_intermediate_features(encoder, x, stage_index):
    """
    Extracts intermediate features from the encoder at the specified stage index.
    """
    if stage_index >= len(encoder.features):
        raise ValueError(
            f"stage_index {stage_index} exceeds the encoder's "
            f"{len(encoder.features)} stages"
        )
    out = x

    for i, block in enumerate(encoder.features):
        out = block(out)
        if i == stage_index:
            return out
    return encoder.to_latent(out)


def build_distance_maps(features_input, features_recon, scope, distance_names, smoothing_sigmas):
    maps = {}

    for distance in distance_names:
        if scope == "position":
            batch_size = features_input.shape[0]
            num_channels = features_input.shape[1]
            spatial_shape = features_input.shape[2:]
            a = features_input.reshape(batch_size, num_channels, -1)
            b = features_recon.reshape(batch_size, num_channels, -1)
            base = pairwise_distance(a, b, distance, dim=1)
            base = base.reshape(batch_size, 1, *spatial_shape)
        elif scope == "element":
            if distance == "l2":
                base = (features_input - features_recon) ** 2
            elif distance == "l1":
                base = (features_input - features_recon).abs()
            else:
                raise ValueError(
                    "Element scope supports l2 and l1 only, got " + str(distance)
                )
        else:
            raise ValueError(
                "build_distance_maps is for position and element scope only"
            )

        for sigma in smoothing_sigmas:
            maps[(distance, sigma)] = smooth_map(base, sigma)

    return maps


def compute_feature_pairs_multi_stage(model, volumes, stage_indices):

    latent = model.encode(volumes)
    reconstruction = model.decoder(latent)

    pairs = {}
    for stage_index in stage_indices:
        if stage_index < 0:
            features_input = latent
            features_recon = model.encode(reconstruction)
        else:
            features_input = get_intermediate_features(
                model.encoder, volumes, stage_index
            )
            features_recon = get_intermediate_features(
                model.encoder, reconstruction, stage_index
            )
        pairs[stage_index] = (features_input.float(), features_recon.float())

    return pairs


def compute_feature_pair(model, volumes, stage_index):

    if stage_index < 0:
        features_input = model.encode(volumes)
        reconstruction = model.decoder(features_input)
        features_recon = model.encode(reconstruction)
    else:
        latent = model.encode(volumes)
        reconstruction = model.decoder(latent)
        features_input = get_intermediate_features(model.encoder, volumes, stage_index)
        features_recon = get_intermediate_features(
            model.encoder, reconstruction, stage_index
        )

    return features_input.float(), features_recon.float()


def pairwise_distance(a, b, distance, dim):

    if distance == "l2":
        return ((a - b) ** 2).sum(dim=dim)

    if distance == "l1":
        return (a - b).abs().sum(dim=dim)

    if distance == "linf":
        return (a - b).abs().max(dim=dim).values

    if distance == "cosine":
        return 1.0 - F.cosine_similarity(a, b, dim=dim, eps=EPSILON)

    if distance == "correlation":
        a_centered = a - a.mean(dim=dim, keepdim=True)
        b_centered = b - b.mean(dim=dim, keepdim=True)
        return 1.0 - F.cosine_similarity(a_centered, b_centered, dim=dim, eps=EPSILON)

    if distance == "relative_l2":
        # Scale invariant like cosine, but a magnitude mismatch still counts.
        difference = torch.sqrt(((a - b) ** 2).sum(dim=dim) + EPSILON)
        norm_a = torch.sqrt((a**2).sum(dim=dim) + EPSILON)
        norm_b = torch.sqrt((b**2).sum(dim=dim) + EPSILON)
        return difference / (norm_a + norm_b + EPSILON)

    raise ValueError("Unknown distance: " + str(distance))


def smooth_map(score_map, smoothing_sigma):

    if smoothing_sigma <= 0:
        return score_map

    smoother = GaussianFilter(spatial_dims=3, sigma=smoothing_sigma).to(
        score_map.device
    )
    return smoother(score_map)


def score_from_features(features_input, features_recon, scope, distance, aggregation="mean", top_k_fraction=0.01, smoothing_sigma=0.0):
    
    batch_size = features_input.shape[0]
    spatial_shape = features_input.shape[2:]
    num_channels = features_input.shape[1]

    if scope == "global":
        a = features_input.reshape(batch_size, -1)
        b = features_recon.reshape(batch_size, -1)
        scores = pairwise_distance(a, b, distance, dim=1)
        if distance in ("l2", "l1"):
            scores = scores / a.shape[1]
        return scores

    if scope == "position":
        # One distance per spatial position, collapsing the channel axis.
        a = features_input.reshape(batch_size, num_channels, -1)
        b = features_recon.reshape(batch_size, num_channels, -1)
        score_map = pairwise_distance(a, b, distance, dim=1)
        score_map = score_map.reshape(batch_size, 1, *spatial_shape)
        score_map = smooth_map(score_map, smoothing_sigma)
        return aggregate_map(score_map, aggregation, top_k_fraction)

    if scope == "element":
        # Every channel at every position treated separately. Only the distances
        # that act elementwise make sense here.
        if distance == "l2":
            score_map = (features_input - features_recon) ** 2
        elif distance == "l1":
            score_map = (features_input - features_recon).abs()
        else:
            raise ValueError(
                "Element scope supports l2 and l1 only, got " + str(distance)
            )
        score_map = smooth_map(score_map, smoothing_sigma)
        return aggregate_map(score_map, aggregation, top_k_fraction)

    raise ValueError("Unknown scope: " + str(scope))


def compute_round_trip_scores(model, volumes, stage_index, distance, top_k_fraction=0.01, scope="global",
    aggregation="mean", smoothing_sigma=0.0):

    features_input, features_recon = compute_feature_pair(model, volumes, stage_index)
    return score_from_features(
        features_input,
        features_recon,
        scope=scope,
        distance=distance,
        aggregation=aggregation,
        top_k_fraction=top_k_fraction,
        smoothing_sigma=smoothing_sigma,
    )
