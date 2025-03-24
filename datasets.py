from typing import Tuple

import numpy as np
from flex.data import Dataset, FedDataDistribution, FedDataset, FedDatasetConfig, LazyIndexable


def get_dataset(dataset: str) -> Tuple[FedDataset, Dataset]:
    match dataset:
        case "celeba":
            return _celeba_non_iid()
        case "cifar_10":
            return _cifar_10_iid()
        case "imagenet":
            return _imagenet()
        case "waterbirds":
            return _waterbirds()
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")


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
        y_data = ([y[1] for y in dataset.y_data] if isinstance(dataset.y_data[0], tuple) else dataset.y_data)

        y_data = LazyIndexable(y_data, len(y_data))
        return Dataset(X_data=dataset.X_data, y_data=y_data)

    flex_dataset = flex_dataset.apply(select_label)
    test_data = select_label(test_data)

    return flex_dataset, test_data


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

    return flex_dataset, test_data


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

    return flex_dataset, test_data


def _waterbirds():
    from utils.waterbirds import WaterbirdsDataset
    train_data = WaterbirdsDataset(train=True)
    test_data = WaterbirdsDataset(train=False)
    flex_dataset = Dataset.from_torchvision_dataset(train_data)
    test_data = Dataset.from_torchvision_dataset(test_data)

    def select_label(dataset: Dataset):
        label_index = 0
        y_data = [y[0] for y in dataset.y_data]
        y_data = LazyIndexable(y_data, len(y_data))
        return Dataset(X_data=dataset.X_data, y_data=y_data)

    flex_dataset = select_label(flex_dataset)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 100

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)
    return flex_dataset, test_data
