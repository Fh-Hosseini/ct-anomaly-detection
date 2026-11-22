import torch
from src.ct_anomaly.evaluation.aggregate import aggregate_map


def compute_quantization_scores(distances, aggregation, top_k_fraction):
    return aggregate_map(distances.float(), aggregation, top_k_fraction)


def compute_code_rarity_scores(indices, code_counts):

    batch_size = indices.shape[0]
    flat = indices.reshape(batch_size, -1)

    log_probabilities = torch.log(code_counts + 1e-10)
    selected = log_probabilities[flat]

    return -selected.mean(dim=1)


def collect_code_usage(model, dataloader, device, prepare_fn, num_codes):

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
