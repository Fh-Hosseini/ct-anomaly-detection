"""
Evaluation metrics for anomaly classification.

Threshold-free metrics:
    AUROC, Average Precision (AP = AUPRC), Specificity @ 95% Sensitivity, Specificity @ 99% Sensitivity

Threshold-based metrics :
    Accuracy, F1, Precision (PPV), NPV, Sensitivity (Recall), Specificity
"""
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

THRESHOLD = 0.5


def auroc(labels, probs):
    # how well does the model rank anolamlies above healthy
    return roc_auc_score(labels, probs)


def average_precision(labels, probs):
    # AP = AUPRC metric: the area under the precision-recall curve.
    # how well the model handles the anolamies specifically
    # better for imballance data
    return average_precision_score(labels, probs)


def specificity_at_sensitivity(labels, probs, target_sensitivity=0.95):
    """
    Finds the specificity at the threshold where sensitivity is closest to target sensitivity.
    if we want to get 95% of real anomalies cases, how many healthy cases get incorrect label
    """
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)

    # tp_rate = sensitivity
    # specificity = 1 - fp_rate
    valid_indices = np.where(tp_rate >= target_sensitivity)[0]

    if len(valid_indices) == 0:
        # No threshold got the target sensitivity.
        return 0.0

    # Among thresholds that reach the target, take the one with the
    # highest specificity (lowest fp rate).
    best_index = valid_indices[np.argmin(fp_rate[valid_indices])]
    specificity = 1 - fp_rate[best_index]
    return specificity


def _predictions_at_threshold(probs, threshold):
    return (np.array(probs) >= threshold).astype(int)


def accuracy(labels, probs, threshold=THRESHOLD):
    # of all predictions, what fraction are correct?
    predictions = _predictions_at_threshold(probs, threshold)
    return accuracy_score(labels, predictions)

def sensitivity(labels, probs, threshold=THRESHOLD):
    # Sensitivity = Recall = true positive rate.
    # of all actual anomaly cases, what fraction did the model correctly catch?
    # High sensitivity = few missed diseases (few fn).
    predictions = _predictions_at_threshold(probs, threshold)
    return recall_score(labels, predictions, zero_division=0)

def precision(labels, probs, threshold=THRESHOLD):
    # Precision = PPV
    # of everything the model predict as anomaly, what fraction really were? 
    # if the model says anomaly, how much should we trust that? 
    predictions = _predictions_at_threshold(probs, threshold)
    return precision_score(labels, predictions, zero_division=0)


def f1(labels, probs, threshold=THRESHOLD):
    # balancing precision and sensitivity together. 
    predictions = _predictions_at_threshold(probs, threshold)
    return f1_score(labels, predictions, zero_division=0)


def specificity(labels, probs, threshold=THRESHOLD):
    # tn rate = tn / (tn + fp).
    # of all actual healthy cases, what fraction did the model correctly leave alone? 
    # High specificity = few false alarms (few fp).
    predictions = _predictions_at_threshold(probs, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def npv(labels, probs, threshold=THRESHOLD):
    # tn / (tn + fn).
    # of everything the model predict as healthy, what fraction really were? 
    # if the model says healthy, how confident can we be?
    predictions = _predictions_at_threshold(probs, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def compute_all_metrics(labels, probs, threshold=THRESHOLD):
    """
    computes all metric above and returns them as a dict
    """
    return {
        "auroc": auroc(labels, probs),
        "average_precision": average_precision(labels, probs),
        "specificity_at_95_sensitivity": specificity_at_sensitivity(labels, probs, 0.95),
        "specificity_at_99_sensitivity": specificity_at_sensitivity(labels, probs, 0.99),
        "accuracy": accuracy(labels, probs, threshold),
        "sensitivity": sensitivity(labels, probs, threshold),
        "specificity": specificity(labels, probs, threshold),
        "ppv": precision(labels, probs, threshold),
        "npv": npv(labels, probs, threshold),
        "f1": f1(labels, probs, threshold),
    }