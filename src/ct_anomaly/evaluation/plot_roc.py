import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from src.ct_anomaly.evaluation.thresholds import find_best_threshold_youden


def plot_roc_curve(labels, probs, save_path, default_threshold=0.5):
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)
    auc_value = roc_auc_score(labels, probs)

    best_threshold, best_sensitivity, best_specificity = find_best_threshold_youden(labels, probs)

    default_index = np.argmin(np.abs(thresholds - default_threshold))

    plt.figure(figsize=(7, 7))
    plt.plot(fp_rate, tp_rate, label=f"ROC curve (AUROC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

    plt.scatter(
        1 - best_specificity, best_sensitivity, color="red", zorder=5,
        label=f"Best balance threshold={best_threshold:.3f}\n"
              f"sensitivity={best_sensitivity:.3f}, specificity={best_specificity:.3f}",
    )
    plt.scatter(
        fp_rate[default_index], tp_rate[default_index], color="blue", zorder=5,
        label=f"Default threshold=0.5\n"
              f"sensitivity={tp_rate[default_index]:.3f}, specificity={1 - fp_rate[default_index]:.3f}",
    )

    plt.xlabel("1 - Specificity (FP Rate)")
    plt.ylabel("Sensitivity (TP Rate)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return best_threshold, best_sensitivity, best_specificity


def plot_roc_curve_multi(labels, probs, save_path, points_dict):
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)
    auc_value = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 8))
    plt.plot(fp_rate, tp_rate, label=f"ROC curve (AUROC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")

    colors = ["red", "blue", "green", "purple", "orange"]
    for (name, threshold), color in zip(points_dict.items(), colors):
        idx = np.argmin(np.abs(thresholds - threshold))
        plt.scatter(
            fp_rate[idx], tp_rate[idx], color=color, zorder=5,
            label=f"{name} (thr={threshold:.3f})\nsens={tp_rate[idx]:.3f}, spec={1 - fp_rate[idx]:.3f}",
        )

    plt.xlabel("1 - Specificity (False Positive Rate)")
    plt.ylabel("Sensitivity (True Positive Rate)")
    plt.title("ROC Curve with Operating Points")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()