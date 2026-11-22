"""
Testing loop for resnet baseline
"""

import torch
from src.ct_anomaly.data.prepare_data import prepare_volumes


def collect_logits(
    model, dataloader, device, use_amp, original_hu_range=None, reclip_hu_range=None
):

    model.eval()
    all_logits = []
    all_labels = []
    all_volume_names = []
    with torch.no_grad():
        for volumes, labels, volume_names in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            volumes = prepare_volumes(
                volumes, original_hu_range, reclip_hu_range, target_shape=None
            )

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(volumes)

            all_logits.append(outputs.float().cpu())
            all_labels.append(labels.cpu())
            all_volume_names.extend(volume_names)

    return torch.cat(all_logits), torch.cat(all_labels), all_volume_names


def test(
    model, dataloader, device, use_amp, original_hu_range=None, reclip_hu_range=None
):

    logits, labels, volume_names = collect_logits(
        model, dataloader, device, use_amp, original_hu_range, reclip_hu_range
    )
    probability_abnormal = torch.softmax(logits, dim=1)[:, 1]
    return (
        labels.numpy().tolist(),
        probability_abnormal.numpy().tolist(),
        volume_names,
    )
