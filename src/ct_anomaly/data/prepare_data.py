from src.ct_anomaly.data.hu_reclip import reclip_hu
from src.ct_anomaly.data.resize import downsample_volume

def prepare_volumes(volumes, original_hu_range, reclip_hu_range, target_shape):
    """
    Prepare volumes for training or validation by re-clipping the HU range and downsampling.
    """
    if reclip_hu_range is not None:
        if original_hu_range is None:
            raise ValueError("reclip_hu_range requires original_hu_range")

        volumes = reclip_hu(
            volumes,
            original_hu_range[0], original_hu_range[1],
            reclip_hu_range[0], reclip_hu_range[1],
        )

    if target_shape is not None:
        volumes = downsample_volume(volumes, target_shape)

    return volumes