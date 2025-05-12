import torchvision
import torch


IVAN_DIR = "/mnt/homeGPU/isevillano/Github/SHIELD-1/data/"


def get_imagenet_data(train=True, data_dir=None, transforms=None):
    if data_dir is None:
        data_dir = IVAN_DIR
    root = f"{data_dir}/imagenet_1k/"
    if train:
        dataset = torchvision.datasets.ImageFolder(
            root=root + "train", transform=transforms
        )

    else:
        dataset = torchvision.datasets.ImageFolder(
            root=root + "val", transform=transforms
        )
    return dataset


def load_tiny_imagenet(train=True, data_dir=None, transforms=None):
    import pickle

    with open(
        "imagenet_subset_train.pkl" if train else "imagenet_subset.pkl", "rb"
    ) as f:
        indices = pickle.load(f)
    dataset = get_imagenet_data(train=train, data_dir=data_dir, transforms=transforms)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset


if __name__ == "__main__":
    import pickle

    dataset = get_imagenet_data(data_dir=IVAN_DIR, train=False)
    print("Reading indices")
    indices = [i for i, target in enumerate(dataset.targets) if target < 10]
    with open("imagenet_subset.pkl", "wb") as f:
        pickle.dump(indices, f)

    dataset = get_imagenet_data(train=True, data_dir=IVAN_DIR, transforms=None)
    print("Reading indices")
    indices = [i for i, target in enumerate(dataset.targets) if target < 10]
    with open("imagenet_subset_train.pkl", "wb") as f:
        pickle.dump(indices, f)

    print("Verifying indices")
    dataset = load_tiny_imagenet(data_dir=IVAN_DIR)
    subset_targets = [dataset.dataset.targets[i] for i in dataset.indices]
    assert all(0 <= target < 10 for target in subset_targets), (
        "Subset targets are not in the range [0, 10)"
    )
    print("Indices verified successfully")
    print("Verifying indices")
    dataset = load_tiny_imagenet(data_dir=IVAN_DIR, train=False)
    subset_targets = [dataset.dataset.targets[i] for i in dataset.indices]
    assert all(0 <= target < 10 for target in subset_targets), (
        "Subset targets are not in the range [0, 10)"
    )
    print("Indices verified successfully")
