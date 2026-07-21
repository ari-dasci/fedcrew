from dataclasses import dataclass
from typing import Callable, Tuple, List

import numpy as np

from flex.data import (
    Dataset,
    FedDataDistribution,
    FedDataset,
    FedDatasetConfig,
    LazyIndexable,
)


@dataclass
class DatasetConfig:
    loader: Callable[[], Tuple[FedDataset, Dataset, List[int]]]


def _celeba_non_iid(label: int = -9):
    import dill as pickle
    from flex.datasets.federated_datasets import federated_celeba

    try:
        with open("celeba_fed.pck", "rb") as f:
            flex_dataset = pickle.load(f)
        with open("celeba_test.pck", "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        print("Creating CelebA dataset")
        flex_dataset, test_data = federated_celeba("..", return_test=True)
        pickle.dump(flex_dataset, open("celeba_fed.pck", "wb"))
        pickle.dump(test_data, open("celeba_test.pck", "wb"))
        print("Done creating CelebA dataset")

    def select_label(dataset: Dataset):
        label_index = label
        assert dataset.y_data is not None, "y_data is None"
        y_data = (
            [y[1] for y in dataset.y_data]
            if isinstance(dataset.y_data[0], tuple)
            else dataset.y_data
        )
        y_data = [y[label_index] for y in y_data]
        y_data = LazyIndexable(y_data, len(y_data))
        return Dataset(X_data=dataset.X_data, y_data=y_data)

    flex_dataset = flex_dataset.apply(select_label)
    test_data = select_label(test_data)
    return flex_dataset, test_data, []


def _cifar_10_iid():
    from torchvision.datasets import CIFAR10

    train_data = CIFAR10(root="../data", train=True, download=True, transform=None)
    test_data = CIFAR10(root="../data", train=False, download=True, transform=None)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    assert isinstance(flex_dataset, FedDataset)
    return flex_dataset, test_data, []


def _get_dirichlet_distribution(n_clients, alpha=0.1, n_classes=10, total_per_class=5000):
    """
    Simulates the class distribution for FedNova/NIID-Bench setup.

    Args:
        n_clients (int): Number of clients.
        alpha (float): Concentration parameter (0.1 = high hetero).
        n_classes (int): Number of classes (10 for Cifar-10, 100 for Cifar-100).
        total_per_class (int): Number of training samples per class.

    Returns:
        dist_matrix (np.ndarray): Shape (n_clients, n_classes).
                                  Entry [i, j] is the ratio of label j in client i.
    """
    cifar_class_cols = []
    for _ in range(n_classes):
        proportions = np.random.dirichlet(np.array([alpha] * n_clients))

        counts = (proportions * total_per_class).astype(int)

        diff = total_per_class - counts.sum()
        if diff != 0:
            idx = np.argmax(proportions)
            counts[idx] += diff

        cifar_class_cols.append(counts)

    client_counts = np.array(cifar_class_cols).T

    client_total_samples = client_counts.sum(axis=1, keepdims=True)

    client_total_samples[client_total_samples == 0] = 1

    dist_matrix = client_counts / client_total_samples

    return dist_matrix


def _cifar_10_non_iid():
    import dill as pickle
    from torchvision.datasets import CIFAR10

    try:
        with open("cifar10_non_iid_fed.pck", "rb") as f:
            flex_dataset = pickle.load(f)
        with open("cifar10_non_iid_test.pck", "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        print("Creating CIFAR-10 non-IID dataset")
        train_data = CIFAR10(root="../data", train=True, download=True, transform=None)
        test_data = CIFAR10(root="../data", train=False, download=True, transform=None)
        flex_dataset = Dataset.from_torchvision_dataset(train_data)
        test_data = Dataset.from_torchvision_dataset(test_data)

        dist_matrix = _get_dirichlet_distribution(
            n_clients=200, alpha=0.1, n_classes=10
        )

        config = FedDatasetConfig(seed=0)
        config.replacement = True
        config.n_nodes = 200
        config.weights_per_label = dist_matrix

        flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

        data_threshold = 30
        cids = list(flex_dataset.keys())
        for k in cids:
            if len(flex_dataset[k]) < data_threshold:
                del flex_dataset[k]

        pickle.dump(flex_dataset, open("cifar10_non_iid_fed.pck", "wb"))
        pickle.dump(test_data, open("cifar10_non_iid_test.pck", "wb"))
        print("Done creating CIFAR-10 non-IID dataset")

    assert isinstance(flex_dataset, FedDataset)
    return flex_dataset, test_data, []


def _cifar_100_non_iid():
    import dill as pickle
    from torchvision.datasets import CIFAR100

    try:
        with open("cifar100_non_iid_fed.pck", "rb") as f:
            flex_dataset = pickle.load(f)
        with open("cifar100_non_iid_test.pck", "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        print("Creating CIFAR-100 non-IID dataset")
        train_data = CIFAR100(root="../data", train=True, download=True, transform=None)
        test_data = CIFAR100(root="../data", train=False, download=True, transform=None)
        flex_dataset = Dataset.from_torchvision_dataset(train_data)
        test_data = Dataset.from_torchvision_dataset(test_data)

        dist_matrix = _get_dirichlet_distribution(
            n_clients=200, alpha=0.1, n_classes=100, total_per_class=500
        )

        config = FedDatasetConfig(seed=0)
        config.replacement = True
        config.n_nodes = 200
        config.weights_per_label = dist_matrix

        flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

        data_threshold = 30
        cids = list(flex_dataset.keys())
        for k in cids:
            if len(flex_dataset[k]) < data_threshold:
                del flex_dataset[k]

        pickle.dump(flex_dataset, open("cifar100_non_iid_fed.pck", "wb"))
        pickle.dump(test_data, open("cifar100_non_iid_test.pck", "wb"))
        print("Done creating CIFAR-100 non-IID dataset")

    assert isinstance(flex_dataset, FedDataset)
    return flex_dataset, test_data, []


def _non_iid_mnist():
    from flex import datasets

    train, test = datasets.load("federated_emnist", return_test=True)
    return train, test, []


# Number of classes for each dataset
DATASET_NUM_CLASSES = {
    "celeba": 2,
    "celeba_a": 2,
    "celeba_m": 2,
    "cifar_10_non_iid": 10,
    "cifar_10": 10,
    "cifar_100_non_iid": 100,
    "mnist_non_iid": 10,
}

DATASET_CONFIG = {
    "celeba": DatasetConfig(loader=_celeba_non_iid),
    "celeba_a": DatasetConfig(loader=lambda: _celeba_non_iid(label=2)),
    "celeba_m": DatasetConfig(loader=lambda: _celeba_non_iid(label=18)),
    "cifar_10_non_iid": DatasetConfig(loader=_cifar_10_non_iid),
    "cifar_10": DatasetConfig(loader=_cifar_10_iid),
    "cifar_100_non_iid": DatasetConfig(loader=_cifar_100_non_iid),
    "mnist_non_iid": DatasetConfig(loader=_non_iid_mnist),
}


def get_dataset(dataset: str) -> Tuple[FedDataset, Dataset, List[int]]:
    config = DATASET_CONFIG.get(dataset)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    return config.loader()


def get_num_classes(dataset: str) -> int:
    """Get the number of classes for a dataset.

    Args:
        dataset: Name of the dataset.

    Returns:
        Number of classes in the dataset.

    Raises:
        ValueError: If the dataset is not recognized.
    """
    if dataset not in DATASET_NUM_CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_NUM_CLASSES[dataset]
