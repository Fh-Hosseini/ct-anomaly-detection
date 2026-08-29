"""
Config for the supervised ResNet baseline.
"""

import os
import json
import glob
from datetime import datetime
import fcntl

from configs.preprocessing_configs import CONFIGS

# PREPROCESSING_CONFIG = "config1"


PREPROCESSING_CONFIG = os.environ.get("PREPROCESSING_CONFIG", "config1")

# RECLIP_HU_MAX = 400  # None: to disable reclipping

_reclip_env = os.environ.get("RECLIP_HU_MAX")
RECLIP_HU_MAX = int(_reclip_env) if _reclip_env else None

LABELS_CSV_PATH = "data/processed/labels_cleaned_with_split.csv"

PREPROCESSED_ROOT = "/anvme/workspace/iwi5437h-ct-anomaly-detection/preprocessed"
PREPROCESSED_DATA_DIR = f"{PREPROCESSED_ROOT}/{CONFIGS[PREPROCESSING_CONFIG]['name']}"

SEED = 42

BATCH_SIZE = 10
NUM_WORKERS = 14
NUM_WORKERS_VAL = 8

NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
USE_AMP = True
USE_CLASS_WEIGHTING = True
EARLY_STOPPING_EPOCHS = 5



LOSS_TYPE = "weighted_ce"   # ce, weighted_ce, focal
FOCAL_GAMMA = 2.0

RESNET_DEPTH = 18
NUM_CLASSES = 2  # healthy, abnormal


RUN_STATE_PATH = "results/.run_state.json"


def _get_experiment_name():
    override = os.environ.get("EXPERIMENT_NAME_OVERRIDE")
    if override:
        return override

    is_new_run = os.environ.get("NEW_RUN") == "1"
    os.makedirs("results", exist_ok=True)
    lock_path = "results/.run_state.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        state_exists = os.path.exists(RUN_STATE_PATH)
        if state_exists and not is_new_run:
            with open(RUN_STATE_PATH) as f:
                return json.load(f)["experiment_name"]
        current_index = 0
        if state_exists:
            with open(RUN_STATE_PATH) as f:
                current_index = json.load(f).get("index", 0)
        next_index = current_index + 1
        reclip_suffix = f"_reclip{RECLIP_HU_MAX}" if RECLIP_HU_MAX else ""
        name = (
            f"{next_index:03d}_resnet{RESNET_DEPTH}_"
            f"prep{PREPROCESSING_CONFIG}_b{BATCH_SIZE}_e{NUM_EPOCHS}_Lr{LEARNING_RATE}"
            f"_wd{WEIGHT_DECAY}_{LOSS_TYPE}{reclip_suffix}"
        )
        with open(RUN_STATE_PATH, "w") as f:
            json.dump({"index": next_index, "experiment_name": name}, f)
        return name

        
EXPERIMENT_NAME = _get_experiment_name()
RUN_DATE = datetime.now().strftime("%Y%m%d")



RESULTS_DIR = f"results/{EXPERIMENT_NAME}"
CONFIG_INFO_PATH = f"{RESULTS_DIR}/config_info.json"
CHECKPOINT_PATH = f"{RESULTS_DIR}/best_checkpoint.pt"
RESUME_CHECKPOINT_PATH = f"{RESULTS_DIR}/resume_checkpoint.pt"
DONE_TRAINING_PATH = f"{RESULTS_DIR}/done.json"
METRICS_LOG_PATH = f"{RESULTS_DIR}/epoch_metrics.jsonl"
TEST_RESULTS_PATH = f"{RESULTS_DIR}/test_results.json"

TEST_PREDICTIONS_PATH = f"{RESULTS_DIR}/test_predictions.csv"
# one row appended here per run.
PROJECT_SUMMARY_PATH = "results/experiments_summary.csv"