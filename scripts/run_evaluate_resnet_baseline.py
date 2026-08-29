"""
Final test-set evaluation for the supervised ResNet baseline.
"""
import os

import json
import pandas as pd

import torch
from torch.utils.data import DataLoader

from configs.resnet_baseline_config import *
from src.ct_anomaly.data.dataset import CTDataset
from src.ct_anomaly.models.resnet3d_monai import resnet3d
from src.ct_anomaly.evaluation.evaluate import test
from src.ct_anomaly.evaluation.metrics import compute_all_metrics
from src.ct_anomaly.evaluation.experiment_log import log_experiment_summary

from src.ct_anomaly.evaluation.plot_roc import plot_roc_curve, plot_roc_curve_multi
from src.ct_anomaly.evaluation.metrics import specificity_at_sensitivity, specificity_at_threshold
from src.ct_anomaly.evaluation.thresholds import *

import numpy as np
from sklearn.metrics import roc_curve

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = os.environ.get("EVAL_EXPERIMENT_NAME") or EXPERIMENT_NAME
    results_dir = f"results/{experiment_name}"
    checkpoint_path = f"{results_dir}/best_checkpoint.pt"
    config_info_path = f"{results_dir}/config_info.json"
    done_training_path = f"{results_dir}/done.json"
    test_results_path = f"{results_dir}/test_results.json"

    with open(config_info_path) as f:
        config_info = json.load(f)

    preprocessing_config = config_info["preprocessing_config"]
    reclip_hu_max = config_info["reclip_hu_max"]

    preprocessed_data_dir = f"{PREPROCESSED_ROOT}/{CONFIGS[preprocessing_config]['name']}"
    original_range = (CONFIGS[preprocessing_config]["hu_min"], CONFIGS[preprocessing_config]["hu_max"])
    reclip_range = (CONFIGS[preprocessing_config]["hu_min"], reclip_hu_max) if reclip_hu_max else None

    test_dataset = CTDataset(LABELS_CSV_PATH, preprocessed_data_dir, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
    )

    val_dataset = CTDataset(LABELS_CSV_PATH, preprocessed_data_dir, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
    )

    model = resnet3d(depth=RESNET_DEPTH, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    val_labels, val_probs, _ = test(model, val_loader, device, USE_AMP,
    original_hu_range=original_range, reclip_hu_range=reclip_range)

    chosen_threshold, val_sens, val_spec = find_best_threshold_youden(val_labels, val_probs)
    print(f"Threshold chosen on VALIDATION: {chosen_threshold:.3f} (val sensitivity={val_sens:.3f}, val specificity={val_spec:.3f})")

    
    thresholds = {
        "youden": find_best_threshold_youden(val_labels, val_probs)[0],
        "sens90": find_threshold_for_sensitivity(val_labels, val_probs, 0.90),
        "sens95": find_threshold_for_sensitivity(val_labels, val_probs, 0.95),
        "sens99": find_threshold_for_sensitivity(val_labels, val_probs, 0.99),
        "cost5to1": find_cost_weighted_threshold(val_labels, val_probs, cost_of_missed_case=5, cost_of_false_alarm=1)[0],
    }

    # print(f"DEBUG: original_range={original_range}, reclip_range={reclip_range}")
    labels, probs, volume_names = test(model, test_loader, device, USE_AMP,
        original_hu_range=original_range, reclip_hu_range=reclip_range)


    test_predictions_path = f"{results_dir}/test_predictions.csv"
    predictions_df = pd.DataFrame({
        "volume_name": volume_names,
        "binary_label": labels,
        "predicted_prob": probs,
    })
    
    predictions_df.to_csv(test_predictions_path, index=False)

    roc_plot_path = f"{results_dir}/roc_curve.png"
    best_threshold, best_sens, best_spec = plot_roc_curve(labels, probs, roc_plot_path)
    plot_roc_curve_multi(labels, probs, f"{results_dir}/roc_curve_all_points.png", thresholds)
    print(f"best threshold: {best_threshold:.3f} (sensitivity={best_sens:.3f}, specificity={best_spec:.3f})")

    results = compute_all_metrics(labels, probs)  # threshold-free metrics only (auroc, ap)


    with open(done_training_path) as f:
        training_info = json.load(f)

    summary_row = dict(config_info)
    summary_row["best_val_auroc"] = training_info["best_val_auroc"]

    for point_name, threshold in thresholds.items():
        if threshold is None:
            print(f"No threshold found for {point_name}")
            continue
        point_metrics = compute_all_metrics(labels, probs, threshold=threshold)
        print(f"\n--- Threshold: {point_name} (threshold={threshold:.4f}) ---")
        for metric_name, value in point_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        for metric_name, value in point_metrics.items():
            results[f"{point_name}_{metric_name}"] = value
        summary_row[f"{point_name}_threshold"] = threshold

    print("Test set results:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    with open(test_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {test_results_path}")

    for metric_name, value in results.items():
        summary_row[f"test_{metric_name}"] = value

    log_experiment_summary(PROJECT_SUMMARY_PATH, summary_row)
    print(f"Added summary to {PROJECT_SUMMARY_PATH}")

if __name__ == "__main__":  
    main()