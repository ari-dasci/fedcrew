import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import select_label_indices


def test_select_label_indices_random_is_seeded():
    labels = [0, 0, 0, 1, 1, 1]

    first = select_label_indices(labels, k=2, selection="random", seed=7)
    second = select_label_indices(labels, k=2, selection="random", seed=7)

    assert first == second
    assert len(first) == 4
