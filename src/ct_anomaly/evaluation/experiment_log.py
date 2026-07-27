"""
Two helpers for keeping track of running experiments for all runs.
"""

import os
import json
import csv


def save_config_info(config_info_path, config_dict):
    os.makedirs(os.path.dirname(config_info_path), exist_ok=True)
    with open(config_info_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def log_experiment_summary(project_summary_path, summary_row):
    file_already_exists = os.path.exists(project_summary_path)

    os.makedirs(os.path.dirname(project_summary_path), exist_ok=True)
    with open(project_summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if not file_already_exists:
            writer.writeheader()
        writer.writerow(summary_row)