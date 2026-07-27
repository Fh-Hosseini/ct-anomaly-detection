import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def find_best_threshold_youden(labels, probs):
    """
    sensitivity + specificity - 1, maximized over all thresholds
    """
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)
    best_index = np.argmax(tp_rate - fp_rate)
    return thresholds[best_index], tp_rate[best_index], 1 - fp_rate[best_index]


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