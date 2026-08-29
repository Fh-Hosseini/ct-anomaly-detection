import sys
import os

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


from configs.resnet_baseline_config import *
from src.ct_anomaly.data.dataset import CTDataset
from src.ct_anomaly.models.resnet3d_monai import resnet3d
from src.ct_anomaly.training.resnet_trainer import train
from src.ct_anomaly.evaluation.experiment_log import save_config_info
from src.ct_anomaly.training.losses import FocalLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(DONE_TRAINING_PATH) and os.environ.get("NEW_RUN") != "1":
        print(f"ERROR: {EXPERIMENT_NAME} already completed. Resubmit with NEW_RUN=1 for a new run.")
        sys.exit(1)


    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    original_range = (CONFIGS[PREPROCESSING_CONFIG]["hu_min"], CONFIGS[PREPROCESSING_CONFIG]["hu_max"])
    reclip_range = (CONFIGS[PREPROCESSING_CONFIG]["hu_min"], RECLIP_HU_MAX) if RECLIP_HU_MAX else None

    train_dataset = CTDataset(LABELS_CSV_PATH, PREPROCESSED_DATA_DIR, split="train")

    val_dataset = CTDataset(LABELS_CSV_PATH, PREPROCESSED_DATA_DIR, split="val")

    data_generator = torch.Generator()
    data_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, generator=data_generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS_VAL, pin_memory=True,
    )

    model = resnet3d(depth=RESNET_DEPTH, num_classes=NUM_CLASSES)
    total_parameters = sum(p.numel() for p in model.parameters())
    

    class_weights_tensor = None
    class_weights_list = None
    if USE_CLASS_WEIGHTING:
        class_counts = train_dataset.labels_df["binary_label"].value_counts().sort_index()
        total_samples = class_counts.sum()
        class_weights = total_samples / (2 * class_counts)
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
        class_weights_list = class_weights.values.tolist()

    if LOSS_TYPE == "focal":
        loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=FOCAL_GAMMA)
    elif LOSS_TYPE == "weighted_ce":
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()


    config_info = {
        "experiment_name": EXPERIMENT_NAME,
        "run_date": RUN_DATE,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "random_seed": SEED,
        "resnet_depth": RESNET_DEPTH,
        "total_parameters": total_parameters,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "num_workers_val": NUM_WORKERS_VAL,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "use_amp": USE_AMP,
        "use_class_weighting": USE_CLASS_WEIGHTING,
        "class_weights": class_weights_list,
        "preprocessed_data_dir": PREPROCESSED_DATA_DIR,
        "weight_decay": WEIGHT_DECAY,
        "loss_type": LOSS_TYPE,
        "focal_gamma": FOCAL_GAMMA if LOSS_TYPE == "focal" else None,
        "early_stopping_epochs": EARLY_STOPPING_EPOCHS,
        "reclip_hu_max": RECLIP_HU_MAX,
    }

    print("=" * 60)
    print("Run configuration:")
    for key, value in config_info.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    save_config_info(CONFIG_INFO_PATH, config_info)

    best_val_auroc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        best_checkpoint_path=CHECKPOINT_PATH,
        resume_checkpoint_path=RESUME_CHECKPOINT_PATH,
        done_training_path=DONE_TRAINING_PATH,
        metrics_log_path=METRICS_LOG_PATH,
        device=device,
        use_amp=USE_AMP,
        loss_fn=loss_fn,
        early_stopping_epochs=EARLY_STOPPING_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        original_hu_range=original_range,
        reclip_hu_range=reclip_range,
    )

    print(f"Training finished. Best val_auroc: {best_val_auroc:.4f}")

    
if __name__ == "__main__":
    main()