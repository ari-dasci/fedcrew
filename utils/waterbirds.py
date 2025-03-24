import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


_WATERBIRDS_DIR = "/mnt/homeGPU/mariogmarq/data/waterbirds/waterbird_complete95_forest2water2/"

class WaterbirdsDataset(Dataset):
    def __init__(self, data_dir=_WATERBIRDS_DIR, train=True):
        self.data = pd.read_csv(data_dir + "metadata.csv")
        if train:
            self.data = self.data[self.data.split != 1]
        else:
            self.data = self.data[self.data.split == 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row["img_filename"]
        img = Image.open(_WATERBIRDS_DIR + image_name)
        return img, [row["y"], row["place"]]

if __name__ == "__main__":
    data = WaterbirdsDataset()
    print(len(data))