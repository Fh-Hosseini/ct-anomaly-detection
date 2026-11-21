# """
# This module contains functions to compute feature scores for anomaly detection using a given model.
# The feature scores are calculated based on the difference between the features of the input volumes and their reconstructions,
# either at the final latent representation or at an intermediate stage of the encoder.
# """
# import torch


# def get_intermediate_features(encoder, x, stage_index):
#     """
#     Extracts intermediate features from the encoder at the specified stage index.
#     """
#     out = x
#     for i, block in enumerate(encoder.features):
#         out = block(out)
#         if i == stage_index:
#             return out
#     return encoder.to_latent(out)


# def compute_round_trip_scores(model, volumes, stage_index, distance, top_k_fraction=0.01):
#     """
#     Computes feature scores for the given volumes using the specified model and distance metric.
#     """
#     if stage_index < 0:
#         features_input = model.encode(volumes)
#         reconstruction = model.decoder(features_input)
#         features_recon = model.encode(reconstruction)
#     else:
#         latent = model.encode(volumes)
#         reconstruction = model.decoder(latent)
#         features_input = get_intermediate_features(model.encoder, volumes, stage_index)
#         features_recon = get_intermediate_features(model.encoder, reconstruction, stage_index)

#     features_input = features_input.float()
#     features_recon = features_recon.float()

#     batch_size = features_input.shape[0]
#     flat_input = features_input.reshape(batch_size, -1)
#     flat_recon = features_recon.reshape(batch_size, -1)

#     if distance == "l2":
#         difference = (flat_input - flat_recon) ** 2
#         scores = difference.mean(dim=1)

#     elif distance == "cosine":
#         similarity = torch.nn.functional.cosine_similarity(flat_input, flat_recon, dim=1)
#         scores = 1.0 - similarity

#     elif distance == "l2_topk":
#         difference = (flat_input - flat_recon) ** 2
#         num_elements = difference.shape[1]
#         num_top = int(round(num_elements * top_k_fraction))
#         if num_top < 1:
#             num_top = 1
#         top_values = torch.topk(difference, k=num_top, dim=1).values
#         scores = top_values.mean(dim=1)

#     elif distance == "l2_max":
#         difference = (flat_input - flat_recon) ** 2
#         scores = difference.max(dim=1).values
#     else:
#         raise ValueError("Unknown distance: " + str(distance))

#     return scores


"""
Round-trip feature scores.

A volume is encoded, decoded, and encoded again. If the model represents the
volume well the two encodings agree; if not, the reconstruction drifts toward
something the model does represent and the second encoding lands elsewhere.
The score is the disagreement between them.

Three choices are independent:

scope        what a single comparison is made between
             "global"   both feature maps flattened, one distance per volume
             "position" channel vector per spatial position, distance collapses
                        the channel axis, leaving a spatial map
             "element"  every element treated separately, leaving a map with
                        the channel axis kept

distance     how two vectors are compared

aggregation  how a map is collapsed to one number, with optional spatial
             smoothing first. Ignored when the scope is global, since there is
             nothing left to aggregate.
"""

import torch
import torch.nn.functional as F
from monai.networks.layers import GaussianFilter

EPSILON = 1e-8


def get_intermediate_features(encoder, x, stage_index):
    """
    Extracts intermediate features from the encoder at the specified stage index.
    """
    out = x
    for i, block in enumerate(encoder.features):
        out = block(out)
        if i == stage_index:
            return out
    return encoder.to_latent(out)

def build_distance_maps(features_input, features_recon, scope, distance_names,
                        smoothing_sigmas):
    """
    Precompute every distance map a batch needs, once per distance and sigma.

    The smoothing is the expensive part and depends only on the distance and the
    sigma, not on the aggregation, so an ablation that tries many aggregations
    per sigma computes each map once here and then only aggregates.

    Returns a dict mapping (distance, sigma) to a map of shape (batch, ...).
    """
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
                raise ValueError("Element scope supports l2 and l1 only, got " + str(distance))
        else:
            raise ValueError("build_distance_maps is for position and element scope only")

        for sigma in smoothing_sigmas:
            maps[(distance, sigma)] = smooth_map(base, sigma)

    return maps


def compute_feature_pairs_multi_stage(model, volumes, stage_indices):
    """
    Encode, decode once, then extract features at several encoder stages.

    The round trip is the expensive part and does not depend on which stage the
    comparison is made at, so a stage sweep shares it rather than repeating it.

    Returns a dict mapping stage index to (features_input, features_recon).
    """
    latent = model.encode(volumes)
    reconstruction = model.decoder(latent)

    pairs = {}
    for stage_index in stage_indices:
        if stage_index < 0:
            features_input = latent
            features_recon = model.encode(reconstruction)
        else:
            features_input = get_intermediate_features(model.encoder, volumes, stage_index)
            features_recon = get_intermediate_features(model.encoder, reconstruction, stage_index)
        pairs[stage_index] = (features_input.float(), features_recon.float())

    return pairs


def compute_feature_pair(model, volumes, stage_index):
    """
    Encode, decode, encode again, and return both feature maps.

    This is the expensive part of the score and does not depend on scope,
    distance or aggregation, so an ablation computes it once per batch and
    reuses it for every variant.

    Returns:
        features_input, features_recon: both (batch, channels, d, h, w)
    """
    if stage_index < 0:
        features_input = model.encode(volumes)
        reconstruction = model.decoder(features_input)
        features_recon = model.encode(reconstruction)
    else:
        latent = model.encode(volumes)
        reconstruction = model.decoder(latent)
        features_input = get_intermediate_features(model.encoder, volumes, stage_index)
        features_recon = get_intermediate_features(model.encoder, reconstruction, stage_index)

    return features_input.float(), features_recon.float()


def pairwise_distance(a, b, distance, dim):
    """
    Distance between two tensors along one axis.

    a, b: tensors of the same shape
    distance: "l2", "l1", "linf", "cosine", "correlation" or "relative_l2"
    dim: axis collapsed by the distance

    Returns a tensor with that axis removed.
    """
    if distance == "l2":
        return ((a - b) ** 2).sum(dim=dim)

    if distance == "l1":
        return (a - b).abs().sum(dim=dim)

    if distance == "linf":
        return (a - b).abs().max(dim=dim).values

    if distance == "cosine":
        return 1.0 - F.cosine_similarity(a, b, dim=dim, eps=EPSILON)

    if distance == "correlation":
        # Cosine after removing each vector's own mean, so a constant offset
        # shared by all channels at a position does not count as agreement.
        a_centered = a - a.mean(dim=dim, keepdim=True)
        b_centered = b - b.mean(dim=dim, keepdim=True)
        return 1.0 - F.cosine_similarity(a_centered, b_centered, dim=dim, eps=EPSILON)

    if distance == "relative_l2":
        # Scale invariant like cosine, but a magnitude mismatch still counts.
        difference = torch.sqrt(((a - b) ** 2).sum(dim=dim) + EPSILON)
        norm_a = torch.sqrt((a ** 2).sum(dim=dim) + EPSILON)
        norm_b = torch.sqrt((b ** 2).sum(dim=dim) + EPSILON)
        return difference / (norm_a + norm_b + EPSILON)

    raise ValueError("Unknown distance: " + str(distance))


def smooth_map(score_map, smoothing_sigma):
    """
    Blur a spatial map of per-position distances.

    score_map: (batch, channels, d, h, w)

    The feature grid is already coarse, so useful sigmas here are far smaller
    than in voxel space. A sigma of one at an 8 cubed latent is roughly an
    eighth of the grid, which is what a sigma near 14 is at 128 cubed.
    """
    if smoothing_sigma <= 0:
        return score_map

    smoother = GaussianFilter(spatial_dims=3, sigma=smoothing_sigma).to(score_map.device)
    return smoother(score_map)


def aggregate_map(score_map, aggregation, top_k_fraction):
    """
    Collapse a per-position or per-element map into one score per volume.

    score_map: (batch, ...) with any trailing shape
    aggregation: "mean", "max" or "topk"
    """
    batch_size = score_map.shape[0]
    flat = score_map.reshape(batch_size, -1)

    if aggregation == "mean":
        return flat.mean(dim=1)

    if aggregation == "max":
        return flat.max(dim=1).values

    if aggregation == "topk":
        num_elements = flat.shape[1]
        num_top = int(round(num_elements * top_k_fraction))
        if num_top < 1:
            num_top = 1
        return torch.topk(flat, k=num_top, dim=1).values.mean(dim=1)

    raise ValueError("Unknown aggregation: " + str(aggregation))


def score_from_features(features_input, features_recon, scope, distance,
                        aggregation="mean", top_k_fraction=0.01,
                        smoothing_sigma=0.0):
    """
    Turn a pair of feature maps into one score per volume.

    Separated from the encoding so that an ablation can evaluate many variants
    on the same pair.
    """
    batch_size = features_input.shape[0]
    spatial_shape = features_input.shape[2:]
    num_channels = features_input.shape[1]

    if scope == "global":
        # One distance for the whole volume. Nothing is left to aggregate, so
        # aggregation and smoothing do not apply.
        a = features_input.reshape(batch_size, -1)
        b = features_recon.reshape(batch_size, -1)
        scores = pairwise_distance(a, b, distance, dim=1)
        if distance in ("l2", "l1"):
            # Keep these comparable to the per-element mean rather than growing
            # with the number of elements.
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
            raise ValueError("Element scope supports l2 and l1 only, got " + str(distance))
        score_map = smooth_map(score_map, smoothing_sigma)
        return aggregate_map(score_map, aggregation, top_k_fraction)

    raise ValueError("Unknown scope: " + str(scope))


def compute_round_trip_scores(model, volumes, stage_index, distance,
                           top_k_fraction=0.01, scope="global",
                           aggregation="mean", smoothing_sigma=0.0):
    """
    Convenience wrapper that encodes and scores in one call.

    The defaults reproduce the original behaviour, so the trainer keeps working
    unchanged: a global distance over the flattened final latent.
    """
    features_input, features_recon = compute_feature_pair(model, volumes, stage_index)
    return score_from_features(
        features_input, features_recon,
        scope=scope,
        distance=distance,
        aggregation=aggregation,
        top_k_fraction=top_k_fraction,
        smoothing_sigma=smoothing_sigma,
    )

