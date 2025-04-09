import copy
from typing import List

import torch
from flex.model import FlexModel
from flex.pool import (
    init_server_model,
    deploy_server_model,
    set_aggregated_weights,
    collect_clients_weights,
    aggregate_weights,
)

from models import get_model


@init_server_model
def build_server_model(dataset: str, l2_factor: float = 0.0):
    server_flex_model = FlexModel()
    server_flex_model["model"] = get_model(dataset)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {"weight_decay": l2_factor}
    return server_flex_model


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    new_flex_model = FlexModel()
    new_flex_model["model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(
        server_flex_model["optimizer_func"]
    )
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(
        server_flex_model["optimizer_kwargs"]
    )
    return new_flex_model


@set_aggregated_weights
def set_aggregated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    dev = aggregated_weights[0].get_device()
    dev = "cpu" if dev == -1 else "cuda" + (f":{dev}" if dev > 0 else "")
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(dev) + new)


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return [
        (weight_dict[name] - server_dict[name].to(dev)).type(torch.float)
        for name in weight_dict
    ]


def clean_up_models(client_model: FlexModel, _):
    import gc

    client_model.clear()
    gc.collect()


@aggregate_weights
def causal_weighted_average(
    weights: List[List[torch.Tensor]], ponderation_tensor: torch.Tensor, bias=False
):
    transposed_weights = list(zip(*weights))
    num_non_final_layers = len(transposed_weights) - (2 if bias else 1)
    aggregated_weights = [
        torch.stack(layer_weights).mean(dim=0)
        for layer_weights in transposed_weights[:num_non_final_layers]
    ]

    # Aggregate the final layer
    if bias:
        # TODO: Implement this
        raise NotImplementedError("Bias is not supported yet")

    # ponderation_tensor has shape (n_labels, n_clients, n_features) all ordered as the same way as weights
    # stacked weights will have shape (n_clients, n_labels, n_features)
    stacked_weights = torch.stack(transposed_weights[num_non_final_layers], dim=0)
    ponderation_tensor = torch.transpose(
        ponderation_tensor, 0, 1
    )  # (n_clients, n_labels, n_features)
    assert (
        stacked_weights.shape == ponderation_tensor.shape
    ), f"Fatal error :(, ponderation_tensor and the weights of last layer doesnt have same shape {ponderation_tensor.shape=} != {stacked_weights.shape=}"

    ponderated_weights = (stacked_weights * ponderation_tensor).sum(dim=0)
    # Normalize
    labels_norms = torch.linalg.vector_norm(ponderated_weights, dim=1)  # (n_labels)
    mean_norm = torch.mean(labels_norms)
    coef = torch.sqrt(mean_norm) / torch.sqrt(labels_norms)
    ponderated_weights = ponderated_weights * coef.unsqueeze(1)

    aggregated_weights.append(ponderated_weights)

    return aggregated_weights

def scalable_softmax(input: torch.Tensor, dim=-1):
    n = input.size(dim=dim)
    return torch.softmax(input * torch.log(n), dim=dim)
