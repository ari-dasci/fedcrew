import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

_WATERBIRDS_DIR = "/home/mariogmarq/waterbirds/waterbird_complete95_forest2water2/"


class WaterbirdsDataset(Dataset):
    def __init__(self, data_dir=_WATERBIRDS_DIR, train=True):
        self.data = pd.read_csv(data_dir + "metadata.csv")
        if train:
            self.data = self.data[self.data.split != 1]
        else:
            self.data = self.data[self.data.split == 1]
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["img_filename"]
        img = Image.open(_WATERBIRDS_DIR + image_name)
        return img, [row["y"], row["place"]]

    def get_partitions(self, n_parts: int = 100):
        return create_partitions(self.data, n_parts)


def create_partitions(df: pd.DataFrame, n_parts: int):
    """
    Create partitions of DataFrame indices into n_parts lists such that:
      - Each partition contains rows with both y values (0 and 1).
      - For each y value in a partition, all rows have the same value of 'place'.

    Parameters:
      df (pd.DataFrame): The input DataFrame with columns "y" and "place" (values 0 or 1).
      n_parts (int): Number of partitions to create.

    Returns:
      partition (List[List]): A list of n_parts lists, each containing indices from the DataFrame.
      partition_details (List[Dict[Tuple[int, int], List[int]]]): A list of dictionaries. Each dictionary maps a tuple
             (place, y) to the list of indices in the corresponding partition that have that (place, y) pair.
    """
    # Group indices by the combination (y, place)
    groups = {(y, p): df[(df['y'] == y) & (df['place'] == p)].index.tolist() for y in [0, 1] for p in [0, 1]}

    # Shuffle indices in each group for randomness
    for key in groups:
        random.shuffle(groups[key])

    # Initialize buckets for each y value separately
    partitions_y0 = [[] for _ in range(n_parts)]
    partitions_y1 = [[] for _ in range(n_parts)]

    # Seed each partition with one index for each y
    for i in range(n_parts):
        # For y=0: select a place (0 or 1) that has available indices
        available_places = [p for p in [0, 1] if groups[(0, p)]]
        if not available_places:
            raise ValueError("Not enough data for y=0")
        chosen_place = random.choice(available_places)
        partitions_y0[i].append(groups[(0, chosen_place)].pop())

        # For y=1:
        available_places = [p for p in [0, 1] if groups[(1, p)]]
        if not available_places:
            raise ValueError("Not enough data for y=1")
        chosen_place = random.choice(available_places)
        partitions_y1[i].append(groups[(1, chosen_place)].pop())

    # Assign remaining indices while preserving unique place for each y in a partition
    # Process y=0:
    for p in [0, 1]:
        while groups[(0, p)]:
            # Valid buckets: either empty for y=0 or already set to a matching place value
            valid_buckets = [i for i in range(n_parts) if
                             (not partitions_y0[i] or df.loc[partitions_y0[i][0], 'place'] == p)]
            if not valid_buckets:
                raise ValueError(f"Cannot assign y=0 element with place {p} without violating condition")
            bucket = random.choice(valid_buckets)
            partitions_y0[bucket].append(groups[(0, p)].pop())

    # Process y=1:
    for p in [0, 1]:
        while groups[(1, p)]:
            valid_buckets = [i for i in range(n_parts) if
                             (not partitions_y1[i] or df.loc[partitions_y1[i][0], 'place'] == p)]
            if not valid_buckets:
                raise ValueError(f"Cannot assign y=1 element with place {p} without violating condition")
            bucket = random.choice(valid_buckets)
            partitions_y1[bucket].append(groups[(1, p)].pop())

    # Combine the y=0 and y=1 partitions to form the final partitions
    partitions = [part0 + part1 for part0, part1 in zip(partitions_y0, partitions_y1)]

    # Optionally, shuffle indices within each partition for additional randomness
    for part in partitions:
        random.shuffle(part)


    partition_details = {(p, y): [] for y in [0, 1] for p in [0, 1]}

    # Iterate over each partition and add the partition index for each (place, y) present.
    for i, part in enumerate(partitions):
        # We'll keep track of which (place, y) combinations have been seen in this partition
        seen = set()
        for idx in part:
            key = (df.loc[idx, 'place'], df.loc[idx, 'y'])
            if key not in seen:
                partition_details[key].append(i)
                seen.add(key)

    return partitions, partition_details
