from typing import List, Optional

import torch


def compute_feddyn_loss(
    model: torch.nn.Module,
    server_model: torch.nn.Module,
    prev_grad: Optional[List[torch.Tensor]],
    alpha: float,
):
    """
    Computes the dynamic regularization terms for FedDyn.

    Args:
        model: The current local model (theta).
        server_model: The global model received from the server (theta^{t-1}).
        prev_grad: A list of tensors representing the historical gradient
                   state for this client (nabla L_k(theta_k^{t-1})).
        alpha: The regularization hyperparameter.
    """
    linear_penalty = 0.0
    quad_penalty = 0.0

    if prev_grad is None:
        prev_grad = [torch.zeros_like(p, device="cuda") for p in model.parameters()]

    for p, s_p, g in zip(model.parameters(), server_model.parameters(), prev_grad):
        # Linear term: - <prev_grad, current_params>
        linear_penalty += torch.sum(p.cuda() * g.cuda())

        # Quadratic term: (alpha / 2) * ||current_params - server_params||^2
        quad_penalty += torch.sum((p.cuda() - s_p.cuda()) ** 2)

    return -linear_penalty + (alpha / 2) * quad_penalty


@torch.no_grad()
def update_local_grad_state(
    model: torch.nn.Module,
    server_model: torch.nn.Module,
    prev_grad: Optional[List[torch.Tensor]],
    alpha: float,
):
    new_grad_state = []
    if prev_grad is None:
        prev_grad = [torch.zeros_like(p, device="cuda") for p in model.parameters()]

    for p, s_p, g in zip(model.parameters(), server_model.parameters(), prev_grad):
        updated_g = g - alpha * (p.cuda() - s_p.cuda())
        new_grad_state.append(updated_g.clone())
    return new_grad_state
