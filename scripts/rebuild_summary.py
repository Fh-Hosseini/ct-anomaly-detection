"""
Rebuilds experiments_summary.csv from scratch, reading each experiment's
own config_info.json / done.json / test_results.json. Collects the FULL
set of fields across all experiments first, so every row uses the same
consistent columns, regardless of when each experiment was run.
"""

import os
import json
import glob
import csv

RESULTS_ROOT = "results"
NEW_SUMMARY_PATH = "results/experiments_summary.csv"

all_rows = []

for results_dir in sorted(glob.glob(f"{RESULTS_ROOT}/[0-9][0-9][0-9]_*")):
    config_info_path = f"{results_dir}/config_info.json"
    done_path = f"{results_dir}/done.json"
    test_results_path = f"{results_dir}/test_results.json"

    if not (os.path.exists(config_info_path) and os.path.exists(done_path) and os.path.exists(test_results_path)):
        print(f"Skipping {results_dir} (not fully finished)")
        continue

    with open(config_info_path) as f:
        config_info = json.load(f)
    with open(done_path) as f:
        training_info = json.load(f)
    with open(test_results_path) as f:
        results = json.load(f)

    row = dict(config_info)
    row["best_val_auroc"] = training_info["best_val_auroc"]
    for metric_name, value in results.items():
        row[f"test_{metric_name}"] = value

    all_rows.append(row)

all_fieldnames = []
for row in all_rows:
    for key in row.keys():
        if key not in all_fieldnames:
            all_fieldnames.append(key)

with open(NEW_SUMMARY_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fieldnames)
    writer.writeheader()
    for row in all_rows:
        writer.writerow(row)

print(f"Wrote {len(all_rows)} rows with {len(all_fieldnames)} columns to {NEW_SUMMARY_PATH}")