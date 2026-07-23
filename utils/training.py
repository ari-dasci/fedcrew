"""Training utilities for federated learning."""

import copy
from contextlib import nullcontext
from typing import Tuple

import numpy as np
import torch
from flex.data import Dataset
from flex.model import FlexModel
from torch.utils.data import DataLoader

from config import ExperimentConfig
from datasets import get_num_classes
from models import get_transforms
from utils.feddyn import compute_feddyn_loss, update_local_grad_state
from utils.fedprox import fedprox_regularization
from utils.moon import get_representation, moon_contrastive_loss


def _autocast(device: str):
    """bf16 autocast on CUDA, no-op elsewhere (keeps CPU-mocked tests exact)."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_dataloader(
    dataset, batch_size: int, shuffle: bool, num_workers: int, device: str = "cuda"
) -> DataLoader:
    """DataLoader with worker/pinning/prefetch tuned when num_workers > 0."""
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )


def train(
    client_flex_model: FlexModel,
    client_data: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> None:
    """Train a client model.

    Args:
        client_flex_model: FlexModel containing the client model and training config.
        client_data: Client's local dataset.
        config: Experiment configuration.
        device: Device to train on.
    """
    data_transforms = get_transforms(config.dataset)
    train_dataset = client_data.to_torchvision_dataset(transform=data_transforms)
    client_dataloader = _make_dataloader(
        train_dataset,
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.dataloader_workers,
        device=device,
    )

    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    autocast_ctx = _autocast(device)

    number_iterations = 0
    for _ in range(config.epochs):
        for images, labels in client_dataloader:
            number_iterations += 1
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast_ctx:
                pred = model(images)
                loss = criterion(pred, labels)

                # L1 regularization on FC layer
                if config.l1 > 0.0:
                    l1_loss = sum(p.abs().sum() for p in model.fc.parameters())
                    loss += config.l1 * l1_loss

                # L2 regularization on FC layer (proximal)
                if config.l2_fc > 0.0:
                    server_fc = client_flex_model["server_model"].fc
                    l2_loss = torch.sum(
                        torch.stack(
                            [
                                ((p - server_p.to(device)).pow(2).sum() / p.numel())
                                for p, server_p in zip(
                                    model.fc.parameters(), server_fc.parameters()
                                )
                            ]
                        )
                    )
                    loss += config.l2_fc * l2_loss

                # FedProx regularization
                if config.fedprox > 0.0:
                    fedprox_loss = fedprox_regularization(
                        model, client_flex_model["server_model"], mu=config.fedprox
                    )
                    loss += fedprox_loss

                # MOON model-contrastive regularization
                if config.moon:
                    server_model = client_flex_model["server_model"].to(device).eval()
                    prev_model = client_flex_model.get("prev_model")
                    z = get_representation(model, images)
                    with torch.no_grad():
                        z_glob = get_representation(server_model, images)
                        z_prev = (
                            get_representation(prev_model.to(device).eval(), images)
                            if prev_model is not None
                            else None
                        )
                    if z_prev is not None:
                        loss += config.moon_mu * moon_contrastive_loss(
                            z, z_glob, z_prev, tau=config.moon_tau
                        )

                # FedDyn dynamic regularization
                if config.feddyn > 0.0:
                    feddyn_grad = client_flex_model.get("feddyn_grad")
                    loss += compute_feddyn_loss(
                        model,
                        client_flex_model["server_model"],
                        feddyn_grad,
                        alpha=config.feddyn,
                    )

            loss.backward()
            optimizer.step()

    # Store iteration count for FedNova
    if config.fednova:
        client_flex_model["fednova_iters"] = number_iterations

    # Store this round's trained model as the client's "previous" model for MOON
    if config.moon:
        client_flex_model["prev_model"] = copy.deepcopy(model).cpu()

    # Update this client's persisted local gradient state for FedDyn
    if config.feddyn > 0.0:
        feddyn_grad = client_flex_model.get("feddyn_grad")
        client_flex_model["feddyn_grad"] = update_local_grad_state(
            model,
            client_flex_model["server_model"],
            feddyn_grad,
            alpha=config.feddyn,
        )


def obtain_metrics(
    server_flex_model: FlexModel,
    test_data: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> Tuple[float, float, np.ndarray]:
    """Compute metrics on test data.

    Args:
        server_flex_model: Model to evaluate.
        test_data: Test dataset.
        config: Experiment configuration.
        device: Device to evaluate on.
        is_server: Whether this is the server model (for logging purposes).

    Returns:
        Tuple of (loss, accuracy, confusion_matrix).
    """
    model = server_flex_model["model"]
    model.eval()
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    autocast_ctx = _autocast(device)

    data_transforms = get_transforms(config.dataset)
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = _make_dataloader(
        test_dataset,
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=config.dataloader_workers,
        device=device,
    )

    num_classes = get_num_classes(config.dataset)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    total_count = torch.zeros((), device=device, dtype=torch.long)
    correct_count = torch.zeros((), device=device, dtype=torch.long)
    loss_sum = torch.zeros((), device=device)
    num_batches = 0

    # Accumulate everything on-device and sync (.item()) exactly once at the
    # end -- per-batch .item()/.cpu() calls each force a blocking GPU sync.
    with torch.inference_mode(), autocast_ctx:
        for data, target in test_dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss_sum += criterion(output, target).float()
            num_batches += 1

            pred = output.argmax(dim=1)
            total_count += target.size(0)
            correct_count += (pred == target).sum()

            # Vectorized confusion-matrix update (was a per-sample Python loop).
            idx = target.long() * num_classes + pred.long()
            confusion_matrix += torch.bincount(
                idx, minlength=num_classes * num_classes
            ).view(num_classes, num_classes)

    test_loss = (loss_sum / num_batches).item()
    test_acc = (correct_count.float() / total_count).item()
    confusion_matrix = confusion_matrix.cpu().numpy()

    return test_loss, test_acc, confusion_matrix


def obtain_metrics_with_predictions(
    server_flex_model: FlexModel,
    test_data: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> Tuple[float, float, np.ndarray, dict]:
    """Compute metrics on test data, also capturing per-sample logits/predictions.

    Unlike `obtain_metrics`, the test dataloader is not shuffled so that the
    returned predictions/targets have a stable, reproducible sample ordering
    that can be joined against `test_data` later for per-class analysis.

    Args:
        server_flex_model: Model to evaluate.
        test_data: Test dataset.
        config: Experiment configuration.
        device: Device to evaluate on.

    Returns:
        Tuple of (loss, accuracy, confusion_matrix, predictions), where
        predictions is a dict with "logits" (FloatTensor[N, num_classes]),
        "preds" (LongTensor[N]), and "targets" (LongTensor[N]).
    """
    model = server_flex_model["model"]
    model.eval()
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    autocast_ctx = _autocast(device)

    data_transforms = get_transforms(config.dataset)
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = _make_dataloader(
        test_dataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=config.dataloader_workers,
        device=device,
    )

    all_logits = []
    all_preds = []
    all_targets = []

    num_classes = get_num_classes(config.dataset)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    total_count = torch.zeros((), device=device, dtype=torch.long)
    correct_count = torch.zeros((), device=device, dtype=torch.long)
    loss_sum = torch.zeros((), device=device)
    num_batches = 0

    with torch.inference_mode(), autocast_ctx:
        for data, target in test_dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss_sum += criterion(output, target).float()
            num_batches += 1

            pred = output.argmax(dim=1)
            total_count += target.size(0)
            correct_count += (pred == target).sum()

            idx = target.long() * num_classes + pred.long()
            confusion_matrix += torch.bincount(
                idx, minlength=num_classes * num_classes
            ).view(num_classes, num_classes)

            # Persisted artifact: cast logits back to fp32 (autocast forward
            # may have produced bf16) before moving off-device.
            all_logits.append(output.float().cpu())
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    test_loss = (loss_sum / num_batches).item()
    test_acc = (correct_count.float() / total_count).item()
    confusion_matrix = confusion_matrix.cpu().numpy()

    predictions = {
        "logits": torch.cat(all_logits, dim=0),
        "preds": torch.cat(all_preds, dim=0),
        "targets": torch.cat(all_targets, dim=0),
    }

    return test_loss, test_acc, confusion_matrix, predictions


def obtain_accuracy(
    server_flex_model: FlexModel,
    test_data: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> float:
    """Compute accuracy on test data.

    Args:
        server_flex_model: Model to evaluate.
        test_data: Test dataset.
        config: Experiment configuration.
        device: Device to evaluate on.

    Returns:
        Test accuracy.
    """
    _, acc, _ = obtain_metrics(server_flex_model, test_data, config, device)
    return acc
