import time
import os
import json

import torch

from src.ct_anomaly.evaluation.metrics import compute_threshold_free_metrics
from src.ct_anomaly.evaluation.voxel_score import compute_voxel_scores
from src.ct_anomaly.data.prepare_data import prepare_volumes
from src.ct_anomaly.evaluation.round_trip_score import compute_round_trip_scores
from src.ct_anomaly.evaluation.vqvae_score import compute_quantization_scores
from src.ct_anomaly.models.model_output import get_reconstruction


def wrap_reconstruction_loss(loss_fn):
    """
    Wrap a reconstruction loss function so that it unpacks the model output before computing the loss.
    Useful for models that return multiple outputs, that only the reconstructed volumes are needed for loss computation.
    """

    def wrapped_loss_fn(model_output, target):
        return loss_fn(get_reconstruction(model_output), target)

    return wrapped_loss_fn


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    scaler,
    device,
    use_amp,
    original_hu_range,
    reclip_hu_range,
    target_shape,
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    num_batches = len(dataloader)
    epoch_start_time = time.time()

    total_reconstruction = 0.0
    total_perplexity = 0.0
    total_kl = 0.0
    per_channel_kl_sum = None

    for batch_idx, (volumes, labels, volume_names) in enumerate(dataloader):
        volumes = volumes.to(device, non_blocking=True)
        volumes = prepare_volumes(
            volumes, original_hu_range, reclip_hu_range, target_shape
        )

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            model_output = model(volumes)
            loss = loss_fn(model_output, volumes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if hasattr(loss_fn, "last_kl_loss"):
            total_kl += loss_fn.last_kl_loss
            total_reconstruction += loss_fn.last_reconstruction_loss

        batch_per_channel = getattr(loss_fn, "last_per_channel_kl", None)
        if batch_per_channel is not None:
            if per_channel_kl_sum is None:
                per_channel_kl_sum = batch_per_channel.clone()
            else:
                per_channel_kl_sum = per_channel_kl_sum + batch_per_channel

        if hasattr(loss_fn, "last_perplexity"):
            total_perplexity += loss_fn.last_perplexity

        if (batch_idx + 1) % 20 == 0:
            running_minutes = (time.time() - epoch_start_time) / 60
            batches_per_minute = (batch_idx + 1) / running_minutes
            running_avg_loss = total_loss / (batch_idx + 1)
            print(
                f"  batch {batch_idx + 1}/{num_batches} | "
                f"time: {running_minutes:.1f} m | "
                f"speed: {batches_per_minute:.2f} b/m | "
                f"loss: {running_avg_loss:.6f}"
            )

    average_loss = total_loss / len(dataloader)
    average_kl = total_kl / len(dataloader)
    average_reconstruction = total_reconstruction / len(dataloader)

    average_perplexity = total_perplexity / len(dataloader)

    average_per_channel_kl = (
        per_channel_kl_sum / len(dataloader) if per_channel_kl_sum is not None else None
    )
    return (
        average_loss,
        average_kl,
        average_reconstruction,
        average_perplexity,
        average_per_channel_kl,
    )


def validate_epoch(
    model,
    dataloader,
    device,
    use_amp,
    original_hu_range,
    reclip_hu_range,
    target_shape,
    aggregation,
    top_k_fraction,
    smoothing_sigma,
    feature_stage_index,
    feature_scope,
    feature_distance,
    feature_aggregation,
    feature_top_k_fraction,
    feature_smoothing_sigma,
):
    """
    Validate the model for one epoch.
    """

    model.eval()

    all_labels = []
    all_scores = []
    all_feature_scores = []
    all_sample_losses = []
    all_quantization_scores = []

    with torch.no_grad():
        for volumes, labels, _ in dataloader:
            volumes = volumes.to(device, non_blocking=True)
            volumes = prepare_volumes(
                volumes, original_hu_range, reclip_hu_range, target_shape
            )

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                model_output = model(volumes)
            reconstructions = get_reconstruction(model_output)

            # Compute the sample wise reconstruction loss (mean squared error) for each volume in the batch
            squared_error = (volumes.float() - reconstructions.float()) ** 2
            batch_size = squared_error.shape[0]
            sample_losses = squared_error.reshape(batch_size, -1).mean(dim=1)

            # Compute the anomaly scores for each volume in the batch
            scores = compute_voxel_scores(
                volumes,
                reconstructions,
                aggregation=aggregation,
                top_k_fraction=top_k_fraction,
                smoothing_sigma=smoothing_sigma,
            )

            # Compute the feature scores for each volume in the batch
            feature_scores = compute_round_trip_scores(
                model,
                volumes,
                stage_index=feature_stage_index,
                scope=feature_scope,
                distance=feature_distance,
                aggregation=feature_aggregation,
                top_k_fraction=feature_top_k_fraction,
                smoothing_sigma=feature_smoothing_sigma,
            )

            # Compute the quantization scores for each volume in the batch if the model output contains quantization information
            if hasattr(model, "quantizer"):
                quantization_scores = compute_quantization_scores(
                    model_output[2],
                    aggregation=aggregation,
                    top_k_fraction=top_k_fraction,
                )
                all_quantization_scores.extend(quantization_scores.cpu().numpy())

            all_feature_scores.extend(feature_scores.cpu().numpy())

            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().numpy())
            all_sample_losses.extend(sample_losses.cpu().numpy())

    healthy_losses = []
    for label, sample_loss in zip(all_labels, all_sample_losses):
        if label == 0:
            healthy_losses.append(sample_loss)

    if len(healthy_losses) == 0:
        raise ValueError("Validation loader contains no healthy volumes.")

    healthy_loss = float(sum(healthy_losses) / len(healthy_losses))
    metrics = compute_threshold_free_metrics(all_labels, all_scores)

    feature_metrics = compute_threshold_free_metrics(all_labels, all_feature_scores)
    for name in feature_metrics:
        metrics["feature_" + name] = feature_metrics[name]

    healthy_feature_scores = [
        score for score, label in zip(all_feature_scores, all_labels) if label == 0
    ]
    metrics["healthy_feature_score_mean"] = float(
        sum(healthy_feature_scores) / len(healthy_feature_scores)
    )

    abnormal_feature_scores = [
        score for score, label in zip(all_feature_scores, all_labels) if label != 0
    ]
    if len(abnormal_feature_scores) > 0:
        metrics["abnormal_feature_score_mean"] = float(
            sum(abnormal_feature_scores) / len(abnormal_feature_scores)
        )

    if len(all_quantization_scores) > 0:
        quantization_metrics = compute_threshold_free_metrics(
            all_labels, all_quantization_scores
        )
        for name in quantization_metrics:
            metrics["quantization_" + name] = quantization_metrics[name]

    return healthy_loss, metrics


def _save_resume_checkpoint(
    resume_checkpoint_path,
    model,
    optimizer,
    scaler,
    epoch,
    best_val_healthy_loss,
    epochs_without_improvement,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_healthy_loss": best_val_healthy_loss,
            "epochs_without_improvement": epochs_without_improvement,
        },
        resume_checkpoint_path,
    )


def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    best_checkpoint_path,
    resume_checkpoint_path,
    done_training_path,
    metrics_log_path,
    device,
    use_amp=True,
    loss_fn=None,
    early_stopping_epochs=8,
    weight_decay=0.0,
    original_hu_range=None,
    reclip_hu_range=None,
    target_shape=None,
    aggregation="topk",
    top_k_fraction=0.01,
    smoothing_sigma=0.0,
    feature_stage_index=-1,
    feature_scope="position",
    feature_distance="l1",
    feature_aggregation="topk",
    feature_top_k_fraction=0.02,
    feature_smoothing_sigma=1.5,
):
    """
    Train the model for a specified number of epochs, with optional early stopping based on validation loss.
    """

    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    start_epoch = 0
    best_val_healthy_loss = float("inf")

    if os.path.exists(done_training_path):
        with open(done_training_path) as f:
            done_training_info = json.load(f)
        print(
            f"Training already completed. Best val_healthy_loss: {done_training_info['best_val_healthy_loss']:.6f}"
        )
        return done_training_info["best_val_healthy_loss"]

    epochs_without_improvement = 0

    if os.path.exists(resume_checkpoint_path):
        checkpoint = torch.load(
            resume_checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_healthy_loss = checkpoint["best_val_healthy_loss"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(
            f"Resuming from epoch {start_epoch} (best_val_healthy_loss: {best_val_healthy_loss:.6f})"
        )

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} is starting:")
        epoch_start_time = time.time()

        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)

        (
            train_loss,
            train_kl,
            train_reconstruction,
            train_perplexity,
            train_per_channel_kl,
        ) = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            scaler,
            device,
            use_amp,
            original_hu_range=original_hu_range,
            reclip_hu_range=reclip_hu_range,
            target_shape=target_shape,
        )

        val_healthy_loss, val_metrics = validate_epoch(
            model,
            val_loader,
            device,
            use_amp,
            original_hu_range=original_hu_range,
            reclip_hu_range=reclip_hu_range,
            target_shape=target_shape,
            aggregation=aggregation,
            top_k_fraction=top_k_fraction,
            smoothing_sigma=smoothing_sigma,
            feature_stage_index=feature_stage_index,
            feature_scope=feature_scope,
            feature_distance=feature_distance,
            feature_aggregation=feature_aggregation,
            feature_top_k_fraction=feature_top_k_fraction,
            feature_smoothing_sigma=feature_smoothing_sigma,
        )

        val_auroc = val_metrics["auroc"]
        epoch_duration_minutes = (time.time() - epoch_start_time) / 60

        perplexity_part = (
            f"perplexity: {train_perplexity:.1f} | " if train_perplexity > 0 else ""
        )
        print(
            f"\nEpoch {epoch + 1}/{num_epochs} | "
            f"train_loss: {train_loss:.6f} | "
            f"val_healthy_loss: {val_healthy_loss:.6f} | "
            f"val_auroc: {val_auroc:.4f} | "
            f"feature_auroc: {val_metrics['feature_auroc']:.4f} | "
            f"{perplexity_part}"
            f"time: {epoch_duration_minutes:.1f} min"
        )

        print("\n  All validation metrics (monitoring only):")
        for metric_name, value in val_metrics.items():
            print(f"    {metric_name}: {value:.4f}")

        metrics_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_kl": train_kl,
            "train_reconstruction": train_reconstruction,
            "val_healthy_loss": val_healthy_loss,
            "train_perplexity": train_perplexity,
            "beta_this_epoch": getattr(loss_fn, "beta", None),
        }

        if train_per_channel_kl is not None:
            metrics_log["active_latent_channels"] = int(
                (train_per_channel_kl >= 0.01).sum()
            )
            metrics_log["total_kl_per_element"] = float(train_per_channel_kl.sum())
            metrics_log["per_channel_kl"] = [
                round(float(v), 5) for v in train_per_channel_kl
            ]

        metrics_log.update(val_metrics)
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(metrics_log) + "\n")

        if val_healthy_loss < best_val_healthy_loss:
            best_val_healthy_loss = val_healthy_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  New best val_healthy_loss: {val_healthy_loss:.6f} saved")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        _save_resume_checkpoint(
            resume_checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            best_val_healthy_loss,
            epochs_without_improvement,
        )

        print("#" * 100)

        if epochs_without_improvement >= early_stopping_epochs:
            print(f"Early stopping: no improvement for {early_stopping_epochs} epochs.")
            break

    with open(done_training_path, "w") as f:
        json.dump({"best_val_healthy_loss": best_val_healthy_loss}, f)

    return best_val_healthy_loss
