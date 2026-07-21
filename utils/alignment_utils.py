"""CKA / classifier-head divergence utilities for feature-alignment analysis.

Quantifies how aligned client representations/heads are under a given
aggregation scheme (FedAvg/FedProx/FedNova/MOON/FedCReW) -- runs regardless of
which aggregator is active, using the fixed anchor subsample as the common
input batch CKA requires.
"""

from typing import Dict

import torch
from flex.data import Dataset
from flex.model import FlexModel
from torch.utils.data import DataLoader

from config import ExperimentConfig
from models import get_relevance_layer, get_transforms
from utils.crp_utils import load_client_model


def get_layer_activations(model, layer_name: str, images: torch.Tensor) -> torch.Tensor:
    """Flattened forward-pass activations at a named layer, for a batch of images."""
    module = dict(model.named_modules())[layer_name]
    captured = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output

    handle = module.register_forward_hook(hook)
    try:
        model(images)
    finally:
        handle.remove()

    activation = captured["activation"]
    return activation.reshape(activation.size(0), -1)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear CKA (Kornblith et al., 2019) between two (n_samples, n_features) matrices."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    xty = x.t() @ y
    xtx = x.t() @ x
    yty = y.t() @ y

    hsic_xy = torch.sum(xty**2)
    hsic_xx = torch.sum(xtx**2)
    hsic_yy = torch.sum(yty**2)

    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-12)


def compute_client_alignment(
    server_flex_model: FlexModel,
    _: Dataset,
    subsamples: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Pairwise CKA (feature alignment) and fc-weight divergence between clients.

    Args:
        server_flex_model: Server model, holding this round's collected
            per-client weight deltas in `"weights"`.
        _: Unused server dataset parameter (for FlexPool.map compatibility).
        subsamples: Fixed anchor subsample, used as the common CKA input batch.
        config: Experiment configuration.
        device: Device to run on.

    Returns:
        Dict with "cka_matrix"/"fc_divergence_matrix" ((n_clients, n_clients))
        and their off-diagonal means.
    """
    base_model = server_flex_model["model"]
    weights = server_flex_model["weights"]
    n_clients = len(weights)

    data_transforms = get_transforms(config.dataset)
    dataset = subsamples.to_torchvision_dataset(transform=data_transforms)
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, _ = next(iter(loader))
    images = images.to(device)

    layer_name = get_relevance_layer(config.dataset)

    activations = []
    fc_weights = []
    for client_id in range(n_clients):
        client_model = load_client_model(base_model, weights, client_id, device)
        client_model.eval()
        with torch.no_grad():
            activations.append(get_layer_activations(client_model, layer_name, images))
        fc = getattr(client_model, "fc")
        fc_weights.append(torch.cat([p.detach().flatten() for p in fc.parameters()]))

    cka_matrix = torch.zeros(n_clients, n_clients)
    fc_divergence_matrix = torch.zeros(n_clients, n_clients)
    for i in range(n_clients):
        for j in range(n_clients):
            cka_matrix[i, j] = linear_cka(activations[i], activations[j])
            fc_divergence_matrix[i, j] = torch.norm(fc_weights[i] - fc_weights[j])

    if n_clients > 1:
        off_diagonal = ~torch.eye(n_clients, dtype=torch.bool)
        cka_mean = cka_matrix[off_diagonal].mean()
        fc_divergence_mean = fc_divergence_matrix[off_diagonal].mean()
    else:
        cka_mean = torch.tensor(1.0)
        fc_divergence_mean = torch.tensor(0.0)

    return {
        "cka_matrix": cka_matrix,
        "cka_mean": cka_mean,
        "fc_divergence_matrix": fc_divergence_matrix,
        "fc_divergence_mean": fc_divergence_mean,
    }
