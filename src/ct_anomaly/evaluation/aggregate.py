import torch


def aggregate_map(score_map, aggregation, top_k_fraction):

    batch_size = score_map.shape[0]
    flat = score_map.reshape(batch_size, -1)
    if aggregation == "mean":
        return flat.mean(dim=1)
    if aggregation == "sum":
        return flat.sum(dim=1)
    if aggregation == "max":
        return flat.max(dim=1).values
    if aggregation == "topk":
        num_top = max(1, int(round(flat.shape[1] * top_k_fraction)))
        return torch.topk(flat, k=num_top, dim=1).values.mean(dim=1)
    raise ValueError("Unknown aggregation: " + str(aggregation))
