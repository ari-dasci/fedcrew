import torch
from torch import nn


def fedprox_regularization(model: nn.Module, server_model: nn.Module, mu=0.01):
    """
    FedProx regularization term for federated learning.

    Args:
        model (nn.Module): The local model of the client.
        server_model (nn.Module): The global model from the server.
        mu (float): The regularization parameter.

    Returns:
        torch.Tensor: The FedProx regularization loss.
    """
    loss = 0.0
    for local_param, global_param in zip(model.parameters(), server_model.parameters()):
        loss += torch.sum((local_param - global_param) ** 2)

    return mu * loss
