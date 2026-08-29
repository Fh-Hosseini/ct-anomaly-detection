"""
Functions for selecting a decision threshold from validation data, to be
applied unchanged to test data (no leakage). Each function answers a
different question about what "best" means.
"""

import numpy as np
from sklearn.metrics import roc_curve


def find_best_threshold_youden(labels, probs):
    """
    Youden's J = sensitivity + specificity - 1, maximized. Ties are
    broken toward higher sensitivity, since our clinical priority
    favors not missing real cases.
    """
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)
    j_scores = tp_rate - fp_rate
    max_j = j_scores.max()
    tied_indices = np.where(j_scores == max_j)[0]
    best_index = tied_indices[np.argmax(tp_rate[tied_indices])]
    return thresholds[best_index], tp_rate[best_index], 1 - fp_rate[best_index]


def find_threshold_for_sensitivity(labels, probs, target_sensitivity):
    """
    Finds the threshold with the best (lowest) false positive rate,
    among thresholds that reach AT LEAST target_sensitivity.
    """
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)
    valid_indices = np.where(tp_rate >= target_sensitivity)[0]
    if len(valid_indices) == 0:
        return None
    return thresholds[valid_indices[np.argmin(fp_rate[valid_indices])]]


def find_cost_weighted_threshold(labels, probs, cost_of_missed_case, cost_of_false_alarm):
    """
    Minimizes total weighted cost: cost_of_missed_case * (missed real
    cases) + cost_of_false_alarm * (false alarms).
    """
    fp_rate, tp_rate, thresholds = roc_curve(labels, probs)
    fn_rate = 1 - tp_rate
    total_cost = (cost_of_missed_case * fn_rate) + (cost_of_false_alarm * fp_rate)
    best_index = np.argmin(total_cost)
    return thresholds[best_index], tp_rate[best_index], 1 - fp_rate[best_index]