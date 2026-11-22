"""
Functions for keeping track of running experiments for all runs.
"""

import os
import json
import csv


def save_config_info(config_info_path, config_dict):
    os.makedirs(os.path.dirname(config_info_path), exist_ok=True)
    with open(config_info_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def log_experiment_summary(project_summary_path, summary_row):

    os.makedirs(os.path.dirname(project_summary_path), exist_ok=True)
    fieldnames = list(summary_row.keys())
    file_already_exists = os.path.exists(project_summary_path)

    if file_already_exists:
        with open(project_summary_path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing and existing != fieldnames:
            missing = set(existing) - set(fieldnames)
            extra = set(fieldnames) - set(existing)
            raise ValueError(
                "summary row does not match the existing CSV header. "
                f"missing: {sorted(missing)}, unexpected: {sorted(extra)}"
            )
        fieldnames = existing

    with open(project_summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_already_exists:
            writer.writeheader()
        writer.writerow(summary_row)
