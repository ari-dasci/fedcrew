import torch
from torch.nn import functional as F


def get_representation(model, images: torch.Tensor) -> torch.Tensor:
    """Penultimate-layer representation, captured as the input to `model.fc`.

    Used as MOON's contrastive representation `z` since this codebase has no
    separate projection head.
    """
    captured = {}

    def hook(module, inputs):
        captured["z"] = inputs[0]

    handle = model.fc.register_forward_pre_hook(hook)
    try:
        model(images)
    finally:
        handle.remove()
    return captured["z"]


def moon_contrastive_loss(
    z: torch.Tensor, z_glob: torch.Tensor, z_prev: torch.Tensor, tau: float = 0.5
) -> torch.Tensor:
    """MOON's model-contrastive loss (Li et al., CVPR 2021, arXiv:2103.16257).

    Pulls `z` toward `z_glob` and away from `z_prev` via a two-way softmax over
    cosine similarities, equivalent to the paper's Eq. 4 with a single negative.
    """
    sim_glob = F.cosine_similarity(z, z_glob) / tau
    sim_prev = F.cosine_similarity(z, z_prev) / tau
    logits = torch.stack([sim_glob, sim_prev], dim=1)
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)
