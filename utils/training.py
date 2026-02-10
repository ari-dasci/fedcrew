"""Training utilities for federated learning."""

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

            loss.backward()
            optimizer.step()

    # Store iteration count for FedNova
    if config.fednova:
        client_flex_model["fednova_iters"] = number_iterations


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
