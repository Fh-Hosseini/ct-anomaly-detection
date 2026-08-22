"""
Training loop for the supervised ResNet baseline.
"""

import time
import os
import json

import torch

from src.ct_anomaly.evaluation.metrics import compute_all_metrics
from src.ct_anomaly.data.hu_reclip import reclip_hu

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, use_amp,
    original_hu_range=None, reclip_hu_range=None):

    model.train()

    total_loss = 0.0
    num_batches = len(dataloader)
    epoch_start_time = time.time()

    for batch_idx, (volumes, labels, _) in enumerate(dataloader):
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if reclip_hu_range is not None:
            volumes = reclip_hu(volumes, original_hu_range[0], original_hu_range[1],
                reclip_hu_range[0], reclip_hu_range[1])

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(volumes)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            running_minutes = (time.time() - epoch_start_time) / 60
            batches_per_minute = (batch_idx + 1) / running_minutes
            running_avg_loss = total_loss / (batch_idx + 1)
            print(
                f"  batch {batch_idx + 1}/{num_batches} | "
                f"time: {running_minutes:.1f} m | "
                f"speed: {batches_per_minute:.2f} b/m | "
                f"loss: {running_avg_loss:.4f}"
            )

    average_loss = total_loss / len(dataloader)
    return average_loss


def validate_epoch(model, dataloader, loss_fn, device, use_amp,
    original_hu_range=None, reclip_hu_range=None):

    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs_unhealthy = []

    with torch.no_grad():
        for volumes, labels, _ in dataloader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            if reclip_hu_range is not None:
                volumes = reclip_hu(volumes, original_hu_range[0], original_hu_range[1],
                    reclip_hu_range[0], reclip_hu_range[1])

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(volumes)
                loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            probs_unhealthy = probs[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs_unhealthy.extend(probs_unhealthy.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(all_labels, all_probs_unhealthy)

    return average_loss, metrics


def _save_resume_checkpoint(resume_checkpoint_path, model, optimizer, scaler, epoch, best_val_auroc, epochs_without_improvement):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_auroc": best_val_auroc,
            "epochs_without_improvement": epochs_without_improvement,
        },
        resume_checkpoint_path,
    )


def train(model, train_loader, val_loader, num_epochs, learning_rate, best_checkpoint_path,
    resume_checkpoint_path, done_training_path, metrics_log_path, device, use_amp=True,
    loss_fn=None, early_stopping_epochs=8, weight_decay=0.0,
    original_hu_range=None, reclip_hu_range=None):

    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
        
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    start_epoch = 0
    best_val_auroc = 0.0

    # If training fully finished in previous job, don't restart.
    if os.path.exists(done_training_path):
        with open(done_training_path) as f:
            done_training_info = json.load(f)
        print(f"Training already completed. Best val_auroc: {done_training_info['best_val_auroc']:.4f}")
        return done_training_info["best_val_auroc"]

    epochs_without_improvement = 0
    # If a resume checkpoint exists, resume from that point.
    if os.path.exists(resume_checkpoint_path):
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auroc = checkpoint["best_val_auroc"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(f"Resuming from epoch {start_epoch} (best_val_auroc: {best_val_auroc:.4f})")


    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} is starting:")
        epoch_start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, use_amp)
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device, use_amp)
        val_auroc = val_metrics["auroc"]

        epoch_duration_minutes = (time.time() - epoch_start_time) / 60

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_auroc: {val_auroc:.4f} | "
            f"time: {epoch_duration_minutes:.1f} min"
        )

        print("\n  All validation metrics:")
        for metric_name, value in val_metrics.items():
            print(f"    {metric_name}: {value:.4f}")

        metrics_log = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        metrics_log.update(val_metrics)
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(metrics_log) + "\n")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  New best val_auroc: {val_auroc:.4f} saved")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        _save_resume_checkpoint(
            resume_checkpoint_path, model, optimizer, scaler, epoch, best_val_auroc, epochs_without_improvement
        )

        print("#" * 100)

        if epochs_without_improvement >= early_stopping_epochs:
            print(f"Early stopping: no improvement for {early_stopping_epochs} epochs.")
            break


    with open(done_training_path, "w") as f:
        json.dump({"best_val_auroc": best_val_auroc}, f)

    return best_val_auroc