"""
Final test-set evaluation for the supervised ResNet baseline.
"""

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
from src.ct_anomaly.evaluation.plot_roc import plot_roc_curve

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_range = (CONFIGS[PREPROCESSING_CONFIG]["hu_min"], CONFIGS[PREPROCESSING_CONFIG]["hu_max"])
    reclip_range = (CONFIGS[PREPROCESSING_CONFIG]["hu_min"], RECLIP_HU_MAX) if RECLIP_HU_MAX else None

    test_dataset = CTDataset(LABELS_CSV_PATH, PREPROCESSED_DATA_DIR, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=True,
    )

    model = resnet3d(depth=RESNET_DEPTH, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)

    print(f"DEBUG: original_range={original_range}, reclip_range={reclip_range}")
    labels, probs, volume_names = test(model, test_loader, device, USE_AMP,
        original_hu_range=original_range, reclip_hu_range=reclip_range)

    predictions_df = pd.DataFrame({
        "volume_name": volume_names,
        "binary_label": labels,
        "predicted_prob": probs,
    })

    predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    roc_plot_path = f"{RESULTS_DIR}/roc_curve.png"
    best_threshold, best_sens, best_spec = plot_roc_curve(labels, probs, roc_plot_path)
    print(f"best threshold: {best_threshold:.3f} (sensitivity={best_sens:.3f}, specificity={best_spec:.3f})")

    results = compute_all_metrics(labels, probs)

    print("Test set results:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {TEST_RESULTS_PATH}")


    with open(CONFIG_INFO_PATH) as f:
        config_info = json.load(f)

    with open(DONE_TRAINING_PATH) as f:
        training_info = json.load(f)

    summary_row = dict(config_info)
    summary_row["best_val_auroc"] = training_info["best_val_auroc"]
    for metric_name, value in results.items():
        summary_row[f"test_{metric_name}"] = value

    log_experiment_summary(PROJECT_SUMMARY_PATH, summary_row)
    print(f"Added summary to {PROJECT_SUMMARY_PATH}")

if __name__ == "__main__":  
    main()