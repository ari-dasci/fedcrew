from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F
import torch

def get_model(dataset: str):
    match dataset:
        case "celeba" | "waterbirds":
            return _get_resnet(num_classes=2)
            # return CelebaNet(num_classes=2)
        case _:
            return _get_resnet(num_classes=10)

def _get_resnet(num_classes: int = 10, pretraining=False):
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)
    return model

def get_relevance_layer(dataset: str):
    match dataset:
        # case "celeba":
        #     return "conv4"
        case _:
            return "layer4.1.conv2"

def get_transforms(dataset: str):
    match dataset:
        case "celeba":
            from torchvision.models import ResNet18_Weights
            # return transforms.Compose(
            #     [
            #         transforms.ToPILImage(),
            #         transforms.Resize((224, 224)),
            #         transforms.Grayscale(),
            #         transforms.ToTensor(),
            #     ]
            # )
            return transforms.Compose([transforms.ToPILImage(), ResNet18_Weights.DEFAULT.transforms()])
        case _:
            from torchvision.models import ResNet18_Weights
            return ResNet18_Weights.DEFAULT.transforms()


class CelebaNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.4)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(32)
        self.drop3 = nn.Dropout(p=0.4)

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.bn4 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout(p=0.4)

        self.fc = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        # One block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop1(F.relu(x))
        # print("First block", x.shape) # (None, 112, 112, 32)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop2(F.relu(x))
        # print("Second block", x.shape) # (None, 56, 56, 32)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop3(F.relu(x))
        # print("Third block", x.shape) # (None, 28, 28, 32)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop4(F.relu(x))
        # print("Fourth block", x.shape) # (None, 14, 14, 32)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x  # self.ac(x)