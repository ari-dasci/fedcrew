from dataclasses import dataclass
from typing import Callable, Tuple, List

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


def _celeba_non_iid():
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
        smiling_index = -9
        assert dataset.y_data is not None, "y_data is None"
        y_data = (
            [y[1] for y in dataset.y_data]
            if isinstance(dataset.y_data[0], tuple)
            else dataset.y_data
        )
        y_data = [y[smiling_index] for y in y_data]
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


def _imagenet():
    from imagenetsubset import load_tiny_imagenet

    train_data = load_tiny_imagenet(train=True)
    test_data = load_tiny_imagenet(train=False)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 50

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    assert isinstance(flex_dataset, FedDataset)
    return flex_dataset, test_data, []


def _waterbirds():
    import dill as pickle
    from utils.waterbirds import WaterbirdsDataset

    try:
        with open("waterbirds_fed.pck", "rb") as f:
            flex_dataset = pickle.load(f)
        with open("waterbirds_test.pck", "rb") as f:
            test_data = pickle.load(f)
        with open("waterbirds_indices.pck", "rb") as f:
            must_have_indices = pickle.load(f)
    except FileNotFoundError:
        print("Creating dataset Waterbirds")
        train_data = WaterbirdsDataset(train=True)
        test_data = WaterbirdsDataset(train=False)
        flex_dataset = Dataset.from_torchvision_dataset(train_data)
        test_data = Dataset.from_torchvision_dataset(test_data)

        partition_indices, partition_details = train_data.get_partitions(100)
        must_have_indices = []
        for v in partition_details.values():
            must_have_indices.append(v[0])

        def select_label(dataset: Dataset):
            y_data = [y[0] for y in dataset.y_data]
            y_data = LazyIndexable(y_data, len(y_data))
            return Dataset(X_data=dataset.X_data, y_data=y_data)

        flex_dataset = select_label(flex_dataset)

        config = FedDatasetConfig(
            indexes_per_node=partition_indices, n_nodes=len(partition_indices), seed=0
        )
        flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

        pickle.dump(flex_dataset, open("waterbirds_fed.pck", "wb"))
        pickle.dump(test_data, open("waterbirds_test.pck", "wb"))
        print("Waterbirds dataset created")

    return flex_dataset, test_data, must_have_indices


def _colored_mnist():
    from utils.colored_mnist import create_flex_colored_mnist_environments

    fed_dataset, test_dataset = create_flex_colored_mnist_environments(num_clients=200)

    # We add the last client as must-have to ensure diversity in the test set
    return fed_dataset, test_dataset, [max(fed_dataset.keys()), min(fed_dataset.keys())]


def _non_iid_mnist():
    from flex import datasets

    train, test = datasets.load("federated_emnist", return_test=True)
    return train, test, []


DATASET_CONFIG = {
    "celeba": DatasetConfig(loader=_celeba_non_iid),
    "cifar_10": DatasetConfig(loader=_cifar_10_iid),
    "imagenet": DatasetConfig(loader=_imagenet),
    "waterbirds": DatasetConfig(loader=_waterbirds),
    "waterbirds_multi": DatasetConfig(loader=_waterbirds),
    "colored_mnist": DatasetConfig(loader=_colored_mnist),
    "mnist_non_iid": DatasetConfig(loader=_non_iid_mnist),
}


def get_dataset(dataset: str) -> Tuple[FedDataset, Dataset, List[int]]:
    config = DATASET_CONFIG.get(dataset)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    return config.loader()
