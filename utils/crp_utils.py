"""CRP (Conditional Relevance Propagation) utilities."""

from copy import deepcopy
from typing import Dict, List, Union

import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify
from flex.data import Dataset
from flex.model import FlexModel
from torch import nn
from torch.utils.data import DataLoader
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

from config import ExperimentConfig
from models import get_relevance_layer, get_transforms
from utils.logging_utils import LoggerState, log_crp_heatmap
from utils.prueba_crp import extract_heatmap


def load_client_model(
    original_model: nn.Module,
    collected_weights: List[List[torch.Tensor]],
    client_id: int,
    device: str = "cuda",
) -> nn.Module:
    """Load weights from a specific client into a model copy.

    Args:
        original_model: Base model architecture.
        collected_weights: List of weight updates from all clients.
        client_id: Index of the client to load.
        device: Device to load model on.

    Returns:
        Copy of original model with client's weights loaded.
    """
    assert client_id < len(collected_weights), (
        f"Client ID out of bounds, {client_id} >= {len(collected_weights)}"
    )

    client_weights = collected_weights[client_id]
    new_model = deepcopy(original_model).to(device)

    with torch.no_grad():
        weight_dict = new_model.state_dict()
        for layer_key, new in zip(weight_dict, client_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(device) + new)

    return new_model


def client_crp(
    client_model: nn.Module,
    client_id: Union[int, str],
    sample_dataset: Dataset,
    config: ExperimentConfig,
    logger: LoggerState,
    round_number: int,
    device: str = "cuda",
) -> None:
    """Generate and log CRP heatmaps for a client model.

    Args:
        client_model: Client's neural network model.
        client_id: Client identifier (int or 'server').
        sample_dataset: Dataset with sample images.
        config: Experiment configuration.
        logger: Logger state for tensorboard/wandb.
        round_number: Current federation round.
        device: Device to run on.
    """
    data = sample_dataset.to_torchvision_dataset()
    data_transforms = get_transforms(config.dataset)

    for sample_id in range(len(data)):
        sample, label = data[sample_id]
        heatmap, _ = extract_heatmap(
            client_model,
            layer=get_relevance_layer(config.dataset),
            transforms=data_transforms,
            sample=sample,
            label=label,
        )
        img = imgify(heatmap, cmap="seismic", symmetric=True, grid=(1, 5))
        log_crp_heatmap(logger, img, client_id, sample_id, round_number)


def create_client_heatmaps(
    server_model: FlexModel,
    _: Dataset,
    subsample_dataset: Dataset,
    config: ExperimentConfig,
    logger: LoggerState,
    round_number: int,
    device: str = "cuda",
) -> None:
    """Create heatmaps for all clients.

    Args:
        server_model: Server model containing weights from all clients.
        _: Unused server dataset parameter (for FlexPool.map compatibility).
        subsample_dataset: Dataset with sample images.
        config: Experiment configuration.
        logger: Logger state for tensorboard/wandb.
        round_number: Current federation round.
        device: Device to run on.
    """
    print("Extracting heatmaps")
    model = server_model["model"].to(device)
    weights = server_model["weights"]

    for client_id in range(len(weights)):
        client_model = load_client_model(model, weights, client_id, device)
        client_crp(
            client_model,
            client_id,
            subsample_dataset,
            config,
            logger,
            round_number,
            device,
        )


def create_server_heatmap(
    server_model: FlexModel,
    _: Dataset,
    subsample_dataset: Dataset,
    config: ExperimentConfig,
    logger: LoggerState,
    round_number: int,
    device: str = "cuda",
) -> None:
    """Create heatmap for the server model.

    Args:
        server_model: Server model.
        _: Unused server dataset parameter (for FlexPool.map compatibility).
        subsample_dataset: Dataset with sample images.
        config: Experiment configuration.
        logger: Logger state for tensorboard/wandb.
        round_number: Current federation round.
        device: Device to run on.
    """
    model = server_model["model"]
    client_crp(
        deepcopy(model).to(device),
        "server",
        subsample_dataset,
        config,
        logger,
        round_number,
        device,
    )


def group_correct_class_probs(
    model: nn.Module,
    sample_dataset: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> Dict[int, List[float]]:
    """Group correct class probabilities by label.

    Args:
        model: Neural network model.
        sample_dataset: Dataset with samples of each class (balanced).
        config: Experiment configuration.
        device: Device to run on.

    Returns:
        Dictionary mapping labels to lists of probabilities.
    """
    data_transforms = get_transforms(config.dataset)
    dataset = sample_dataset.to_torchvision_dataset(transform=data_transforms)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=config.batchsize)
    logits: Dict[int, List[float]] = {}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_logits = model(inputs)
            probs = torch.softmax(sample_logits, dim=1)
            correct_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            for label, prob in zip(
                labels.cpu().numpy(), correct_class_probs.cpu().numpy()
            ):
                label = int(label)
                if label not in logits:
                    logits[label] = []
                logits[label].append(float(prob))

    # If any sample in a class has prob < 0.5, zero out all probs for that class
    for label in logits:
        if any(prob < 0.5 for prob in logits[label]):
            logits[label] = [0.0] * len(logits[label])

    return logits


def get_crp_attribution(
    model: nn.Module,
    sample_dataset: Dataset,
    layer: str,
    config: ExperimentConfig,
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """Compute CRP attributions for a model on sample data.

    Args:
        model: Neural network model.
        sample_dataset: Dataset with samples.
        layer: Layer name to compute attributions for.
        config: Experiment configuration.
        device: Device to run on.

    Returns:
        Dictionary mapping labels to attribution tensors.
    """
    data_transforms = get_transforms(config.dataset)
    dataset = sample_dataset.to_torchvision_dataset(transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=1)

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    attribution = CondAttribution(model)

    contributions: dict = {}

    for sample, label in dataloader:
        sample = data_transforms(sample.squeeze()).to(device)
        sample = sample.unsqueeze(0)
        sample.requires_grad = True

        conditions = [{"y": [label]}]
        attr = attribution(sample, conditions, composite, record_layer=layer_names)
        rel_c = cc.attribute(attr.relevances[layer], abs_norm=True)
        rel_c = rel_c[0]
        rel_c_min = torch.min(rel_c)
        rel_c_max = torch.max(rel_c)
        rel_c = (rel_c - rel_c_min) / (rel_c_max - rel_c_min)

        if isinstance(label, torch.Tensor):
            label = label.item()
        if label not in contributions:
            contributions[label] = []
        contributions[label].append(rel_c)

    # Stack contributions per label
    result: Dict[int, torch.Tensor] = {
        label: torch.stack(contributions[label]) for label in contributions
    }

    return result


def compute_features_weights_per_client(
    server_model: FlexModel,
    _: Dataset,
    subsamples: Dataset,
    config: ExperimentConfig,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute feature weights per client using CRP.

    Args:
        server_model: Server model containing client weights.
        _: Unused server dataset parameter (for FlexPool.map compatibility).
        subsamples: Dataset with sample images for CRP.
        config: Experiment configuration.
        device: Device to run on.

    Returns:
        Tensor of shape (n_labels, n_clients, n_features) with feature weights.
    """
    print("Running round CRP")
    model = server_model["model"]
    weights = server_model["weights"]
    feature_dim = model.fc.weight.shape[1]

    clients_probs: Dict[int, Dict[int, List[float]]] = {}
    clients_relevances: Dict[int, Dict[int, torch.Tensor]] = {}

    use_crp = config.causal_crp
    use_logits = config.causal_logits

    for client_id in range(len(weights)):
        client_model = load_client_model(model, weights, client_id, device)
        if use_logits:
            clients_probs[client_id] = group_correct_class_probs(
                client_model, subsamples, config, device
            )
        if use_crp:
            clients_relevances[client_id] = get_crp_attribution(
                client_model,
                subsamples,
                get_relevance_layer(config.dataset),
                config,
                device,
            )

    # We assume that all labels are present in `subsamples`
    label_source = clients_probs if use_logits else clients_relevances
    labels = sorted(list(label_source[0].keys()))
    client_features_weights: Dict[int, torch.Tensor] = {}

    for label in labels:
        if use_logits and use_crp:
            label_probs = torch.tensor(
                [clients_probs[client_id][label] for client_id in range(len(weights))],
                dtype=torch.float32,
            ).to(device)

            label_probs = torch.where(
                label_probs > config.alpha,
                label_probs,
                torch.finfo().min,
            )

            sample_weight = torch.softmax(label_probs.flatten(), dim=0).view(
                *label_probs.shape
            )

            label_relevances = torch.stack(
                [clients_relevances[client_id][label] for client_id in range(len(weights))]
            ).to(device)

            weights_features_clients = torch.sum(
                label_relevances * sample_weight.unsqueeze(-1), dim=1
            )
        elif use_logits:
            label_probs = torch.tensor(
                [clients_probs[client_id][label] for client_id in range(len(weights))],
                dtype=torch.float32,
            ).to(device)
            label_probs = label_probs.mean(dim=1, keepdim=True)
            weights_features_clients = label_probs.expand(-1, feature_dim)
        else:
            label_relevances = torch.stack(
                [clients_relevances[client_id][label] for client_id in range(len(weights))]
            ).to(device)
            weights_features_clients = label_relevances.mean(dim=1)

        client_features_weights[label] = weights_features_clients

    return torch.stack(
        [client_features_weights[label] for label in labels], dim=0
    )  # (n_labels, n_clients, n_features)
