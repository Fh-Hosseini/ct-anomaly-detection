import os
import json
import glob

from src.ct_anomaly.evaluation.experiment_log import log_experiment_summary

RESULTS_ROOT = "results"
NEW_SUMMARY_PATH = "results/experiments_summary.csv"

for results_dir in sorted(glob.glob(f"{RESULTS_ROOT}/[0-9][0-9][0-9]_*")):
    config_info_path = f"{results_dir}/config_info.json"
    done_path = f"{results_dir}/done.json"
    test_results_path = f"{results_dir}/test_results.json"


    if not (os.path.exists(config_info_path) and os.path.exists(done_path) and os.path.exists(test_results_path)):
        print(f"Skipping {results_dir} (not fully finished yet)")
        continue

    with open(config_info_path) as f:
        config_info = json.load(f)
    with open(done_path) as f:
        training_info = json.load(f)
    with open(test_results_path) as f:
        results = json.load(f)

    summary_row = dict(config_info)
    summary_row["best_val_auroc"] = training_info["best_val_auroc"]
    for metric_name, value in results.items():
        summary_row[f"test_{metric_name}"] = value

    log_experiment_summary(NEW_SUMMARY_PATH, summary_row)
    print(f"Added {results_dir}")