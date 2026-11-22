import torch

from src.ct_anomaly.data.prepare_data import prepare_volumes
from src.ct_anomaly.evaluation.voxel_score import compute_voxel_scores
from src.ct_anomaly.evaluation.round_trip_score import compute_round_trip_scores
from src.ct_anomaly.models.model_output import get_reconstruction


def collect_scores(model, dataloader, device, use_amp, original_hu_range, reclip_hu_range, target_shape, score_type,
    aggregation, top_k_fraction, smoothing_sigma, feature_stage_index, feature_scope, feature_distance,
    feature_aggregation, feature_top_k_fraction, feature_smoothing_sigma):
    """
    Collects anomaly scores for the given model and dataloader.
    """
    model.eval()

    all_labels = []
    all_scores = []
    all_volume_names = []

    with torch.no_grad():
        for volumes, labels, volume_names in dataloader:
            volumes = volumes.to(device, non_blocking=True)
            volumes = prepare_volumes(
                volumes, original_hu_range, reclip_hu_range, target_shape
            )

            if score_type == "feature":
                scores = compute_round_trip_scores(
                    model,
                    volumes,
                    stage_index=feature_stage_index,
                    scope=feature_scope,
                    distance=feature_distance,
                    aggregation=feature_aggregation,
                    top_k_fraction=feature_top_k_fraction,
                    smoothing_sigma=feature_smoothing_sigma,
                )
            elif score_type == "voxel":
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    reconstructions = get_reconstruction(model(volumes))
                scores = compute_voxel_scores(
                    volumes,
                    reconstructions,
                    aggregation=aggregation,
                    top_k_fraction=top_k_fraction,
                    smoothing_sigma=smoothing_sigma,
                )
            else:
                raise ValueError("Unknown score_type: " + str(score_type))

            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().numpy())
            all_volume_names.extend(volume_names)

    return all_labels, all_scores, all_volume_names
