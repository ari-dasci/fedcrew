"""Data utilities for federated learning."""

from typing import Iterable, Tuple, cast

import numpy as np
import torch
from flex.data import Dataset, LazyIndexable
from torchvision import transforms

from config import ExperimentConfig
from utils.logging_utils import LoggerState, log_samples


def select_subsample_server_data(
    _,
    dataset: Dataset,
    config: ExperimentConfig,
    logger: LoggerState,
    round_number: int,
    k: int = 2,
) -> Tuple[Dataset, Dataset]:
    """Select k samples per class from the server dataset.

    Args:
        _: Unused server flex_model parameter (for FlexPool.map compatibility).
        dataset: Server dataset.
        config: Experiment configuration.
        logger: Logger state for logging samples.
        round_number: Current federation round.
        k: Number of samples per class to select.

    Returns:
        Tuple of (selected_subsample, remaining_test_dataset).
    """
    print("Creating subsample")
    assert dataset.y_data is not None, "dataset.y_data cannot be None"
    labels = list(cast(Iterable, dataset.y_data))

    if config.dataset == "waterbirds":
        labels = [(label[0], label[1]) for label in labels]

    if config.dataset == "colored_mnist":
        data = dataset.X_data
        assert data is not None, "dataset.X_data cannot be None"
        labels = [
            (label, torch.argmax(img.sum(dim=(1, 2))).item())
            for label, img in zip(labels, cast(Iterable, data))
        ]

    labels_to_indices: dict = {label: [] for label in labels}

    for i, label in enumerate(labels):
        if len(labels_to_indices[label]) < k:
            labels_to_indices[label].append(i)

    indices = [v for v in labels_to_indices.values()]
    indices = [i for sublist in indices for i in sublist]
    new_dataset: Dataset = dataset[indices]

    # Remove indices from the original dataset
    indices_not_included = np.arange(len(dataset))
    indices_not_included = np.setdiff1d(indices_not_included, np.array(indices))
    assert dataset.X_data is not None, "dataset.X_data cannot be None"
    assert dataset.y_data is not None, "dataset.y_data cannot be None"
    new_x_data = LazyIndexable(
        cast(Iterable, dataset.X_data),
        indices_not_included.shape[0],
        indices_not_included,
    )
    new_y_data = LazyIndexable(
        cast(Iterable, dataset.y_data),
        indices_not_included.shape[0],
        indices_not_included,
    )
    new_test_dataset = Dataset(X_data=new_x_data, y_data=new_y_data)

    # Log samples
    torch_data = new_dataset.to_torchvision_dataset()
    transform = transforms.ToTensor()
    samples = []
    for i in range(len(torch_data)):
        sample = transform(torch_data[i][0])
        sample = np.copy(sample.cpu().numpy())
        if sample.shape[0] == 1 or sample.shape[0] == 3:
            sample = np.transpose(sample, (1, 2, 0))
        samples.append(transform(sample))

    log_samples(logger, samples, round_number)

    return new_dataset, new_test_dataset
