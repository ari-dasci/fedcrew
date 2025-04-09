from typing import Optional, Callable

import torch
import numpy as np
import torchvision.datasets
from crp.image import imgify
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
from torchvision import models
from matplotlib import pyplot as plt
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm


def extract_heatmap(model: torch.nn.Module, sample, layer: str, transforms: Optional[Callable], label: int, k=5):
    # Canonizers for CNNs
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    attribution = CondAttribution(model)

    if transforms is not None:
        sample = transforms(sample)
    # extract device from model
    device = next(model.parameters()).get_device()
    device = f"cuda:{device}" if device >= 0 else "cpu"

    sample = sample.to(device).reshape(1, *sample.shape)
    sample.requires_grad = True
    conditions = [{"y": [label]}]
    attr = attribution(sample, conditions, composite, record_layer=layer_names)
    rel_c = cc.attribute(attr.relevances[layer], abs_norm=True)
    if k is None:
        rel_indices = torch.argsort(rel_c[0])
        rel_values = rel_c[0][rel_indices]
    else:
        rel_values, rel_indices = torch.topk(rel_c[0], k=k)
    conditions = [{"y": label, layer: [feat_id]} for feat_id in rel_indices]
    heatmap, _, _, _ = attribution(sample, conditions, composite) # (k, img_size, img_size)
    return heatmap, len(rel_indices)


if __name__ == '__main__':
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Set model to evaluation mode
    model.eval()

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True)
    img_1, y_1 = trainset[20]


    preprocess_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256)), transforms.CenterCrop((224, 224))])
    plt.imsave('original_image.png', preprocess_transform(img_1).cpu().numpy().transpose(1, 2, 0))
    final_transforms = transforms.Compose([preprocess_transform, ResNet18_Weights.DEFAULT.data_transforms()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    heatmap, length = extract_heatmap(model, img_1, 'layer4.1.conv2', final_transforms, k=5, label=y_1)
    grid_image = imgify(heatmap, symmetric=True, grid=(1, length), cmap='seismic')
    plt.imsave('crp.png', np.array(grid_image))
