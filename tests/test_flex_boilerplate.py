import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.flex_boilerplate import clean_up_models


def test_clean_up_models_preserves_prev_model_for_moon():
    client_model = {
        "model": object(),
        "server_model": object(),
        "prev_model": "sentinel-prev-model",
    }

    clean_up_models(client_model, None)

    assert client_model == {"prev_model": "sentinel-prev-model"}


def test_clean_up_models_clears_fully_when_no_prev_model():
    client_model = {"model": object(), "server_model": object()}

    clean_up_models(client_model, None)

    assert client_model == {}
