import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

TEST_PREDICTIONS_PATH = "results/004_resnet18_prepconfig1_b10_e30_Lr0.0001_wd0.0001/test_predictions.csv"
CT_RATE_LABELS_PATHS = [
    "/anvme/workspace/b180dc29-CT_RATE_IDEA_MIRROR/CT-RATE_multi_abnormality_labels/train_predicted_labels.csv",
    "/anvme/workspace/b180dc29-CT_RATE_IDEA_MIRROR/CT-RATE_multi_abnormality_labels/valid_predicted_labels.csv",
]


predictions_df = pd.read_csv(TEST_PREDICTIONS_PATH)


false_positive_rate, true_positive_rate, thresholds = roc_curve(
    predictions_df["binary_label"], predictions_df["predicted_prob"]
)


valid_indices = np.where(true_positive_rate >= 0.99)[0]

best_index = valid_indices[np.argmin(false_positive_rate[valid_indices])]
threshold_99 = thresholds[best_index]
print(f"Threshold for 99% sensitivity: {threshold_99:.4f}")


false_positives = predictions_df[
    (predictions_df["binary_label"] == 0) & (predictions_df["predicted_prob"] >= threshold_99)
]
print(f"False positives at this threshold: {len(false_positives)}")


ct_rate_labels = pd.concat([pd.read_csv(p) for p in CT_RATE_LABELS_PATHS])
ct_rate_labels = ct_rate_labels.rename(columns={"VolumeName": "volume_name"})


merged = false_positives.merge(ct_rate_labels, on="volume_name", how="left")


LUNG_SPECIFIC = [
    "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

has_lung_finding = (merged[LUNG_SPECIFIC].sum(axis=1) > 0)
print(f"False positives with a genuine LUNG finding: {has_lung_finding.sum()} / {len(merged)}")

lung_finding_counts = merged[LUNG_SPECIFIC].sum().sort_values(ascending=False)
print("\nMost common LUNG findings among false-positive 'healthy' volumes:")
print(lung_finding_counts[lung_finding_counts > 0])