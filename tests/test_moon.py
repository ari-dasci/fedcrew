import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.moon import get_representation, moon_contrastive_loss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_get_representation_returns_fc_input_and_removes_hook():
    model = TinyModel()
    images = torch.randn(3, 4)

    z = get_representation(model, images)

    assert torch.equal(z, images)
    assert len(model.fc._forward_pre_hooks) == 0

    # Calling again should still work identically (hook not left registered twice)
    z2 = get_representation(model, images)
    assert torch.equal(z2, images)


def test_moon_contrastive_loss_prefers_closer_representation_to_glob():
    z = torch.tensor([[1.0, 0.0]])
    z_glob_close = torch.tensor([[1.0, 0.0]])
    z_prev_far = torch.tensor([[-1.0, 0.0]])

    low_loss = moon_contrastive_loss(z, z_glob_close, z_prev_far, tau=0.5)

    z_glob_far = torch.tensor([[-1.0, 0.0]])
    z_prev_close = torch.tensor([[1.0, 0.0]])

    high_loss = moon_contrastive_loss(z, z_glob_far, z_prev_close, tau=0.5)

    assert low_loss.item() < high_loss.item()
