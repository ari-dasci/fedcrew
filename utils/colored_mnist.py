# colored_mnist.py
import math
import random
from typing import List, Tuple, Optional
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from flex.data import Dataset as FlexDataset
from flex.data import FedDataDistribution, FedDatasetConfig, FedDataset

# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Colored MNIST Dataset
# -------------------------
class ColoredMNIST(Dataset):
    """
    Colored MNIST as in Arjovsky et al. (IRM paper).

    For each original MNIST example (grayscale 28x28):
      - compute y_tilde = 0 if digit in [0..4], else 1
      - flip y_tilde with prob label_flip_prob (0.25 in paper) -> y
      - sample color z by flipping y with prob p_e (environment-specific)
      - color image: if z==1 -> red, else -> green
    Returns: (colored_image_tensor, y, z, original_digit)
    colored_image_tensor shape: (3, 28, 28), values in [0,1] float32
    """
    def __init__(
        self,
        mnist_split: str = "train",
        env_p: float = 0.1,
        label_flip_prob: float = 0,
        max_samples: Optional[int] = None,
        download: bool = True,
        root: str = "./data",
        seed: int = 0,
        transform: Optional[transforms.Compose] = None,
        lazy_coloring: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        mnist_split: "train" or "test"
        env_p: p_e (probability to flip color w.r.t. label)
        label_flip_prob: probability to flip y_tilde to create noisy label y (default 0.25)
        max_samples: optionally cap number of samples (useful for small experiments)
        lazy_coloring: if True, apply coloring at __getitem__ time; if False, precompute at init
        cache_dir: directory to cache metadata; if None, caching is disabled
        """
        assert mnist_split in ("train", "test")
        self.env_p = float(env_p)
        self.label_flip_prob = float(label_flip_prob)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.transform = transform
        self.lazy_coloring = lazy_coloring
        self.cache_dir = cache_dir

        # load MNIST (grayscale PIL images)
        base_transform = transforms.ToTensor()  # outputs C=1, [0,1]
        self.mnist = datasets.MNIST(root=root, train=(mnist_split=="train"),
                                     download=download, transform=base_transform)
        # optionally cap
        if max_samples is not None and max_samples < len(self.mnist):
            indices = list(range(len(self.mnist)))
            rng = np.random.RandomState(seed)
            chosen = rng.choice(indices, size=max_samples, replace=False)
            self.mnist_data = [self.mnist[i] for i in chosen]
        else:
            self.mnist_data = [self.mnist[i] for i in range(len(self.mnist))]

        # Precompute colored dataset deterministically from seed (or store metadata for lazy coloring)
        self._prepare_coloring()

    def _get_cache_path(self) -> str:
        """Generate a unique cache file path based on dataset parameters."""
        cache_name = (
            f"colored_mnist_"
            f"split={self.mnist.train}_"
            f"env_p={self.env_p}_"
            f"label_flip={self.label_flip_prob}_"
            f"max_samples={self.max_samples}_"
            f"seed={self.seed}_"
            f"lazy_coloring={self.lazy_coloring}.pkl"
        )
        return os.path.join(self.cache_dir, cache_name)

    def _load_cache(self) -> Optional[List]:
        """Load samples from cache if it exists. Returns None if cache miss."""
        if self.cache_dir is None:
            return None
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                import dill
                with open(cache_path, "rb") as f:
                    samples = dill.load(f)
                print(f"[ColoredMNIST] Loaded {len(samples)} samples from cache: {cache_path}")
                return samples
            except Exception as e:
                print(f"[ColoredMNIST] Failed to load cache: {e}")
                return None
        return None

    def _save_cache(self, samples: List):
        """Save samples to cache."""
        if self.cache_dir is None:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_path()
        try:
            import dill
            with open(cache_path, "wb") as f:
                dill.dump(samples, f)
            print(f"[ColoredMNIST] Saved {len(samples)} samples to cache: {cache_path}")
        except Exception as e:
            print(f"[ColoredMNIST] Failed to save cache: {e}")

    def _prepare_coloring(self):
        # Try to load from cache first
        cached_samples = self._load_cache()
        if cached_samples is not None:
            self.samples = cached_samples
            return
        
        # Otherwise, generate samples
        rng = np.random.RandomState(self.seed)
        self.samples = []
        for idx, (img_tensor, digit) in enumerate(self.mnist_data):
            # img_tensor shape (1, 28, 28), values [0,1]
            # original digit
            y_tilde = 0 if (digit <= 4) else 1
            # flip label with probability label_flip_prob
            if rng.rand() < self.label_flip_prob:
                y = 1 - y_tilde
            else:
                y = y_tilde
            # color id z: flip y with prob env_p
            if rng.rand() < self.env_p:
                z = 1 - y
            else:
                z = y
            
            if self.lazy_coloring:
                # Store raw image tensor and metadata; coloring happens at __getitem__
                self.samples.append((img_tensor, y, z, digit))
            else:
                # Precompute colored image (original behavior)
                colored = self._color_sample(img_tensor, z)
                self.samples.append((colored, int(y), int(z), int(digit)))
        
        # Save to cache after generating
        self._save_cache(self.samples)

    def _color_sample(self, img_tensor, z):
        """
        Apply coloring to a grayscale image tensor.
        Returns colored image as numpy array (3, 28, 28).
        """
        g = img_tensor[0].numpy()  # shape (28,28)
        if z == 1:
            r = g
            gr = np.zeros_like(g)
        else:
            r = np.zeros_like(g)
            gr = g
        b = np.zeros_like(g)
        colored = np.stack([r, gr, b], axis=0).astype(np.float32)  # (3,28,28)
        return colored

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.lazy_coloring:
            img_tensor, y, z, digit = self.samples[idx]
            colored = self._color_sample(img_tensor, z)
            img = torch.from_numpy(colored)
        else:
            img_np, y, z, digit = self.samples[idx]
            img = torch.from_numpy(img_np)  # float32 tensor
        
        if self.transform is not None:
            # transform expects PIL or Tensor depending on transform composition.
            # user can pass torchvision.transforms that accept tensors.
            img = self.transform(img)
        return img, y, z, digit


# -------------------------
# Helpers to make environments
# -------------------------
def make_colored_mnist_environments(
    pes: List[float] = [0.2, 0.1, 0.9],
    label_flip_prob: float = 0,
    train_samples_per_env: Optional[int] = None,
    test_samples_per_env: Optional[int] = None,
    root: str = "./data",
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> Tuple[List[ColoredMNIST], ColoredMNIST]:
    """
    Create multiple ColoredMNIST datasets (one per p_e in pes).
    By default pes = [0.2, 0.1, 0.9] (paper: two training envs, one test env).
    Returns: (train_envs_list, test_env)
      - train_envs_list: list for pes[:-1]
      - test_env: dataset for pes[-1]
    
    cache_dir: directory to cache dataset metadata; if None, caching is disabled
    """
    seed_everything(seed)
    datasets_list = []
    for i, p in enumerate(pes):
        # choose split: use mnist train for all envs except possibly the last if user wants test from MNIST test
        # The paper used three environments derived from MNIST - they don't specify mixing train/test split.
        # We'll use MNIST train for all envs for training envs and MNIST test for the test env for a clean separation.
        if i < len(pes) - 1:
            split = "train"
            max_samples = train_samples_per_env
        else:
            split = "test"
            max_samples = test_samples_per_env
        ds = ColoredMNIST(
            mnist_split=split,
            env_p=p,
            label_flip_prob=label_flip_prob,
            max_samples=max_samples,
            download=True,
            root=root,
            seed=seed + i,  # different seed per environment for randomness decorrelation
            transform=None,
            lazy_coloring=True,
            cache_dir=cache_dir,
        )
        datasets_list.append(ds)
    train_envs = datasets_list[:-1]
    test_env = datasets_list[-1]
    return train_envs, test_env


def create_flex_colored_mnist_environments(
    num_clients: int = 200,
    label_flip_prob: float = 0,
    test_samples_per_env: Optional[int] = 2000,
    root: str = "./data",
    seed: int = 0,
    cache_dir: Optional[str] = "./colored_mnist_cache",
) -> Tuple[FedDataset, FlexDataset]:
    """
    Create multiple ColoredMNIST datasets (one per p_e in pes) as Flex datasets.
    By default pes = [0.2, 0.1, 0.9] (paper: two training envs, one test env).
    Returns: (train_envs_flex_dataset, test_env_flex_dataset)
      - train_envs_flex_dataset: FedDataDi for pes[:-1]
      - test_env_flex_dataset: FlexDataset for pes[-1]
    
    cache_dir: directory to cache dataset metadata; if None, caching is disabled
    """
    train_samples_per_env = 60_000 // (num_clients)
    seed_everything(seed)

    highly_correlated_clients = int(0.9 * num_clients)
    pes_highly_correlated = [0.1] * highly_correlated_clients
    pes = pes_highly_correlated + [0.3] * (num_clients - highly_correlated_clients) + [0.9]
    
    train_envs, test_env = make_colored_mnist_environments(
        pes=pes,
        label_flip_prob=label_flip_prob,
        train_samples_per_env=train_samples_per_env,
        test_samples_per_env=test_samples_per_env,
        root=root,
        seed=seed,
        cache_dir=cache_dir,
    )

    flex_train_envs = []
    for i, ds in enumerate(train_envs):
        flex_ds = FlexDataset.from_torchvision_dataset(ds)
        flex_train_envs.append(flex_ds)
    
    fed_data_dist = FedDataset({i: flex_train_envs[i] for i in range(len(flex_train_envs))})

    flex_test_env = FlexDataset.from_torchvision_dataset(test_env)

    return fed_data_dist, flex_test_env

# -------------------------
# Quick example / sanity check
# -------------------------
if __name__ == "__main__":
    # Example usage: create two training envs and one test env (default pes)
    train_envs, test_env = make_colored_mnist_environments(
        pes=[0.2, 0.1, 0.9],
        label_flip_prob=0.25,
        train_samples_per_env=20000,  # cap for speed; set None to use full MNIST train (60000)
        test_samples_per_env=10000,    # cap for speed; set None to use full MNIST test (10000)
        root="./data",
        seed=0,
        cache_dir="./cache/colored_mnist",  # enable caching
    )

    # Compute and print color/label correlations for the two training envs and test env
    def env_stats(ds: ColoredMNIST):
        n = len(ds)
        ys = np.array([s[1] for s in ds.samples])
        zs = np.array([s[2] for s in ds.samples])
        digits = np.array([s[3] for s in ds.samples])
        # fraction z == y
        frac_z_eq_y = float((zs == ys).mean())
        # fraction colored-red among y==1 and y==0
        frac_red_y1 = float(((zs==1) & (ys==1)).sum() / max(1, (ys==1).sum()))
        frac_red_y0 = float(((zs==1) & (ys==0)).sum() / max(1, (ys==0).sum()))
        return {
            "n": n,
            "frac_z_eq_y": frac_z_eq_y,
            "frac_red_y1": frac_red_y1,
            "frac_red_y0": frac_red_y0,
            "y_mean": ys.mean(),
        }

    print("Environment stats (paper default pes = [0.2,0.1,0.9]):")
    for i, ds in enumerate(train_envs):
        st = env_stats(ds)
        print(f" Train env {i} (p_e={ds.env_p}): n={st['n']}, frac_z_eq_y={st['frac_z_eq_y']:.3f}, "
              f"red|y=1={st['frac_red_y1']:.3f}, red|y=0={st['frac_red_y0']:.3f}, y_mean={st['y_mean']:.3f}")
    st = env_stats(test_env)
    print(f" Test env (p_e={test_env.env_p}): n={st['n']}, frac_z_eq_y={st['frac_z_eq_y']:.3f}, "
          f"red|y=1={st['frac_red_y1']:.3f}, red|y=0={st['frac_red_y0']:.3f}, y_mean={st['y_mean']:.3f}")

    # Example: small DataLoader
    dl = DataLoader(train_envs[0], batch_size=64, shuffle=True)
    imgs, ys, zs, digits = next(iter(dl))
    print("Batch shapes:", imgs.shape, ys.shape, zs.shape)  # imgs (B,3,28,28)
