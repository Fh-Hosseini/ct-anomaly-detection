import torch


def reclip_hu(volume_tensor, original_hu_min, original_hu_max, new_hu_min, new_hu_max):
    hu = (volume_tensor + 1) / 2 * (original_hu_max - original_hu_min) + original_hu_min
    hu_clipped = torch.clamp(hu, new_hu_min, new_hu_max)
    return 2 * (hu_clipped - new_hu_min) / (new_hu_max - new_hu_min) - 1