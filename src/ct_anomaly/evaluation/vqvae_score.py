"""
Anomaly scores that use the discrete bottleneck of a VQ-VAE.

The codebook is learned from healthy volumes, so it comes to describe the range
of normal anatomy. A latent position whose encoding sits far from every code is
evidence of structure the model has no representation for. That distance is
available without decoding at all, which makes it a fundamentally different
signal from reconstruction error: it measures whether the model can represent
the input, not whether it can redraw it.

The distances live on the latent grid, one per position, so they are aggregated
in the same way the voxelwise error map is.
"""

import torch


def compute_quantization_scores(distances, aggregation, top_k_fraction):
    """
    Aggregate per-position quantization distances into one score per volume.

    Args:
        distances: tensor of shape (batch, 1, d, h, w) from the quantizer,
            holding the squared distance from each encoding to its chosen code
        aggregation: "mean", "max" or "topk"
        top_k_fraction: fraction of positions used when aggregation is "topk"

    Returns:
        Tensor of shape (batch,)

    No smoothing is applied. The latent grid has only a few hundred positions,
    so blurring it would remove most of the spatial detail there is.
    """
    distances = distances.float()

    batch_size = distances.shape[0]
    flat = distances.reshape(batch_size, -1)

    if aggregation == "mean":
        return flat.mean(dim=1)

    if aggregation == "max":
        return flat.max(dim=1).values

    if aggregation == "topk":
        num_positions = flat.shape[1]
        num_top = int(round(num_positions * top_k_fraction))
        if num_top < 1:
            num_top = 1
        top_values = torch.topk(flat, k=num_top, dim=1).values
        return top_values.mean(dim=1)

    raise ValueError("Unknown aggregation: " + str(aggregation))


def compute_code_rarity_scores(indices, code_counts):
    """
    Score a volume by how unusual its chosen codes are among healthy volumes.

    Quantization distance asks whether a code fits well. This asks a different
    question: whether the codes chosen are ones healthy anatomy commonly uses. A
    volume built from codes that are rare in the healthy training set is unusual
    even if each code fits closely.

    Args:
        indices: chosen code index per position, shape (batch, d, h, w)
        code_counts: tensor of length num_codes holding how often each code was
            chosen across the healthy training volumes, as a probability

    Returns:
        Tensor of shape (batch,), the mean negative log probability of the
        chosen codes, so that larger means more unusual.
    """
    batch_size = indices.shape[0]
    flat = indices.reshape(batch_size, -1)

    log_probabilities = torch.log(code_counts + 1e-10)
    selected = log_probabilities[flat]

    return -selected.mean(dim=1)


def collect_code_usage(model, dataloader, device, prepare_fn, num_codes):
    """
    Count how often each code is chosen across a dataloader.

    Run on the healthy training volumes, this gives the reference distribution
    that compute_code_rarity_scores needs.
    """
    model.eval()
    counts = torch.zeros(num_codes, device=device)

    with torch.no_grad():
        for volumes, _, _ in dataloader:
            volumes = volumes.to(device, non_blocking=True)
            volumes = prepare_fn(volumes)

            latent = model.encoder(volumes)
            _, _, _, _, indices = model.quantizer(latent)

            flat = indices.reshape(-1)
            counts += torch.bincount(flat, minlength=num_codes).float()

    total = counts.sum()
    return counts / total
