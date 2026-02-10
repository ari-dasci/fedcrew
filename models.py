from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class DatasetConfig:
    num_classes: int
    transforms: Callable[[], transforms.Compose]
    model_factory: Callable[[], nn.Module]


def _get_resnet(num_classes: int = 10, pretraining=False):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)
    return model


DATASET_CONFIG = {
    "celeba": DatasetConfig(
        num_classes=2,
        transforms=lambda: transforms.Compose(
            [transforms.ToPILImage(), ResNet18_Weights.DEFAULT.transforms()]
        ),
        model_factory=lambda: _get_resnet(num_classes=2),
    ),
    "celeba_a": DatasetConfig(
        num_classes=2,
        transforms=lambda: transforms.Compose(
            [transforms.ToPILImage(), ResNet18_Weights.DEFAULT.transforms()]
        ),
        model_factory=lambda: _get_resnet(num_classes=2),
    ),
    "celeba_m": DatasetConfig(
        num_classes=2,
        transforms=lambda: transforms.Compose(
            [transforms.ToPILImage(), ResNet18_Weights.DEFAULT.transforms()]
        ),
        model_factory=lambda: _get_resnet(num_classes=2),
    ),
    "cifar_10_non_iid": DatasetConfig(
        num_classes=10,
        transforms=lambda: ResNet18_Weights.DEFAULT.transforms(),
        model_factory=lambda: _get_resnet(num_classes=10),
    ),
    "mnist_non_iid": DatasetConfig(
        num_classes=10,
        transforms=lambda: transforms.Compose([transforms.ToTensor()]),
        model_factory=lambda: MNISTNet(num_classes=10),
    ),
    "default": DatasetConfig(
        num_classes=10,
        transforms=lambda: ResNet18_Weights.DEFAULT.transforms(),
        model_factory=lambda: _get_resnet(num_classes=10),
    ),
}


def fetch_relevance_layer(model: nn.Module) -> str:
    if isinstance(model, MNISTNet):
        return "fc_mid"
    return "layer4.1.conv2"  # Resnet-18


def get_model(dataset: str):
    config = DATASET_CONFIG.get(dataset, DATASET_CONFIG["default"])
    return config.model_factory()


def get_transforms(dataset: str):
    config = DATASET_CONFIG.get(dataset, DATASET_CONFIG["default"])
    return config.transforms()


def get_relevance_layer(dataset: str):
    config = DATASET_CONFIG.get(dataset, DATASET_CONFIG["default"])
    return fetch_relevance_layer(config.model_factory())


class MNISTNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same")
        self.drop1 = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.drop2 = nn.Dropout(p=0.4)
        self.fc_mid = nn.Linear(64 * 7 * 7, 128, bias=False)
        self.fc = nn.Linear(128, num_classes, bias=False)

    def forward(self, x):
        x = self.drop1(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_mid(x))
        return self.fc(x)
