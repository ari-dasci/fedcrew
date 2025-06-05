from flex.model import FlexModel
from scipy import special


def get_fednova_iters(model: FlexModel, _):
    """Extracts the FedNova iteration count from the model."""
    return model.get("fednova_iters", 0)


def obtain_fednova_weights(fednova_iters: list):
    return special.softmax(fednova_iters)
