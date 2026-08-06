"""
Testing loop for resnet baseline
"""

import torch
def collect_logits(model, dataloader, device, use_amp):
    model.eval()

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for volumes, labels, volume_names in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(volumes)

            all_labels.append(labels.cpu())
            all_logits.append(outputs.float().cpu())

    return torch.cat(all_logits), torch.cat(all_labels)

def test(model, dataloader, device, use_amp):
    model.eval()

    all_labels = []
    all_probs_unhealthy = []
    all_volume_names = []

    with torch.no_grad():
        for volumes, labels, volume_names in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(volumes)

            probs = torch.softmax(outputs, dim=1)
            probs_unhealthy = probs[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs_unhealthy.extend(probs_unhealthy.cpu().numpy())
            all_volume_names.extend(volume_names)

    return all_labels, all_probs_unhealthy, all_volume_names