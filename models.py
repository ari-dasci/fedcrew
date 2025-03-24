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
    relevance_layer: str
    model_factory: Callable[[], nn.Module]


def _get_resnet(num_classes: int = 10, pretraining=False):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)
    return model


DATASET_CONFIG = {"celeba": DatasetConfig(num_classes=2, transforms=lambda: transforms.Compose(
    [transforms.ToPILImage(), ResNet18_Weights.DEFAULT.transforms()]), relevance_layer="conv4",
                                          model_factory=lambda: _get_resnet(num_classes=2)),
                  "waterbirds": DatasetConfig(num_classes=2, transforms=lambda: ResNet18_Weights.DEFAULT.transforms(),
                                              relevance_layer="layer4.1.conv2",
                                              model_factory=lambda: _get_resnet(num_classes=2)),
                  "default": DatasetConfig(num_classes=10, transforms=lambda: ResNet18_Weights.DEFAULT.transforms(),
                                           relevance_layer="layer4.1.conv2",
                                           model_factory=lambda: _get_resnet(num_classes=10))}


def get_model(dataset: str):
    config = DATASET_CONFIG.get(dataset, DATASET_CONFIG["default"])
    return config.model_factory()


def get_transforms(dataset: str):
    config = DATASET_CONFIG.get(dataset, DATASET_CONFIG["default"])
    return config.transforms()


def get_relevance_layer(dataset: str):
    config = DATASET_CONFIG.get(dataset, DATASET_CONFIG["default"])
    return config.relevance_layer


class CelebaNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(p=0.4)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(32)
        self.drop3 = nn.Dropout(p=0.4)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout(p=0.4)
        self.fc = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        return self.fc(x)
