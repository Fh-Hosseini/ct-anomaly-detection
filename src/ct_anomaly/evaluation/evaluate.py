"""
Testing loop for resnet baseline
"""

import torch


def test(model, dataloader, device, use_amp):
    model.eval()

    all_labels = []
    all_probs_unhealthy = []

    with torch.no_grad():
        for volumes, labels in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(volumes)

            probs = torch.softmax(outputs, dim=1)
            probs_unhealthy = probs[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs_unhealthy.extend(probs_unhealthy.cpu().numpy())

    return all_labels, all_probs_unhealthy