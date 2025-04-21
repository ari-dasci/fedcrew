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
    Create partitions of DataFrame indices into n_parts lists such that each partition:
      - Is balanced in terms of the label y (i.e., equal number of y=0 and y=1 samples, allowing reuse of minority samples).
      - Contains only pictures from the same "place".

    Parameters:
      df (pd.DataFrame): The input DataFrame with columns "y" and "place".
      n_parts (int): Number of partitions to create.

    Returns:
      partitions (List[List[int]]): A list of n_parts lists, each containing DataFrame indices.
      partition_details (Dict[Tuple[int, int], List[int]]): A dictionary mapping (place, y) to the list of 
             partition indices that contain that (place, y) pair.
    """
    import random
    groups={(y,p):df[(df['y']==y)&(df['place']==p)].index.tolist() for y in [0,1] for p in [0,1]}
    groups_original={key:lst.copy() for key,lst in groups.items()}
    for key in groups:
        random.shuffle(groups[key])
    partitions=[[] for _ in range(n_parts)]
    partition_places=[None]*n_parts
    # Seed each partition with one index per label from the same chosen place.
    for i in range(n_parts):
        candidate_places=[p for p in [0,1] if groups_original[(0,p)] and groups_original[(1,p)]]
        if not candidate_places:
            raise ValueError("Not enough data to seed partition with consistent place for both y values")
        chosen_place=random.choice(candidate_places)
        partition_places[i]=chosen_place
        for y in [0,1]:
            if groups[(y,chosen_place)]:
                partitions[i].append(groups[(y,chosen_place)].pop())
            else:
                partitions[i].append(random.choice(groups_original[(y,chosen_place)]))
    # Evenly add additional balanced pairs for each place.
    for p in [0,1]:
        part_ids=[i for i,v in enumerate(partition_places) if v==p]
        if not part_ids:
            continue
        allowed0=(len(groups_original[(0,p)])//len(part_ids))-1
        allowed1=(len(groups_original[(1,p)])//len(part_ids))-1
        extra_needed=max(min(allowed0,allowed1),0)
        for i in part_ids:
            for _ in range(extra_needed):
                for y in [0,1]:
                    if groups[(y,p)]:
                        partitions[i].append(groups[(y,p)].pop())
                    else:
                        partitions[i].append(random.choice(groups_original[(y,p)]))
    for part in partitions:
        random.shuffle(part)
    partition_details={(p,y):[] for y in [0,1] for p in [0,1]}
    for i,part in enumerate(partitions):
        seen=set()
        for idx in part:
            key=(df.loc[idx,'place'], df.loc[idx,'y'])
            if key not in seen:
                partition_details[key].append(i)
                seen.add(key)
    return partitions, partition_details


if __name__ == "__main__":
    data = WaterbirdsDataset()
    print(len(data))
