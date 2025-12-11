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
    "waterbirds": DatasetConfig(
        num_classes=2,
        transforms=lambda: ResNet18_Weights.DEFAULT.transforms(),
        model_factory=lambda: _get_resnet(num_classes=2),
    ),
    "waterbirds_multi": DatasetConfig(
        num_classes=2,
        transforms=lambda: ResNet18_Weights.DEFAULT.transforms(),
        model_factory=lambda: AutoencoderMultitask(num_classes=2),
    ),
    "colored_mnist": DatasetConfig(
        num_classes=2,
        transforms=lambda: transforms.Compose([]),
        model_factory=lambda: MNISTNet(num_classes=2),
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
    if isinstance(model, CelebaNet):
        return "conv4"
    if isinstance(model, AutoencoderMultitask):
        return "encoder.12"
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


class AutoencoderMultitask(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Codificador
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )

        # Decodificador
        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

        self.fc = nn.Linear(256 * 14 * 14, num_classes, bias=False)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        y = torch.flatten(encoded, 1)
        y = self.fc(y)
        return decoded, y
