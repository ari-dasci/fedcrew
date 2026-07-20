"""Training utilities for federated learning."""

import copy
from typing import Tuple

import numpy as np
import torch
from flex.data import Dataset
from flex.model import FlexModel
from torch.utils.data import DataLoader

from config import ExperimentConfig
from datasets import get_num_classes
from models import get_transforms
from utils.fedprox import fedprox_regularization
from utils.moon import get_representation, moon_contrastive_loss


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
    client_dataloader = DataLoader(
        train_dataset, batch_size=config.batchsize, shuffle=True
    )

    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]

    number_iterations = 0
    for _ in range(config.epochs):
        for images, labels in client_dataloader:
            number_iterations += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
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

            loss.backward()
            optimizer.step()

    # Store iteration count for FedNova
    if config.fednova:
        client_flex_model["fednova_iters"] = number_iterations

    # Store this round's trained model as the client's "previous" model for MOON
    if config.moon:
        client_flex_model["prev_model"] = copy.deepcopy(model).cpu()


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
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]

    data_transforms = get_transforms(config.dataset)
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batchsize, shuffle=True, pin_memory=False
    )

    losses = []

    num_classes = get_num_classes(config.dataset)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

            # Update confusion matrix
            for t, p in zip(target.cpu().view(-1), pred.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix.cpu().numpy()
    test_loss = sum(losses) / len(losses)
    test_acc /= total_count

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
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]

    data_transforms = get_transforms(config.dataset)
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batchsize, shuffle=False, pin_memory=False
    )

    losses = []
    all_logits = []
    all_preds = []
    all_targets = []

    num_classes = get_num_classes(config.dataset)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

            # Update confusion matrix
            for t, p in zip(target.cpu().view(-1), pred.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            all_logits.append(output.cpu())
            all_preds.append(pred.view(-1).cpu())
            all_targets.append(target.cpu())

    confusion_matrix = confusion_matrix.cpu().numpy()
    test_loss = sum(losses) / len(losses)
    test_acc /= total_count

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
