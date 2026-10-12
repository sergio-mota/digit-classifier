import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class TrainDataset(Dataset):
    def __init__(self, data_dir: str, transform: v2.Compose = None):
        self.transform = transform

        filepath = os.path.join(data_dir, "train.pt")
        if os.path.exists(filepath):
            data = torch.load(filepath)
            self.features = data["features"]
            self.labels = data["labels"]
        else:
            try:
                df = pd.read_csv(os.path.join(data_dir, "train.csv"))

                self.features = (
                    torch.tensor(
                        df.drop(columns=["label"]).values.astype("float32")
                    ) / 255.0
                ).reshape(-1, 1, 28, 28)
                self.labels = torch.tensor(df["label"].values, dtype=torch.int64)
                torch.save({"features": self.features, "labels": self.labels}, filepath)
                print(f"Saved processed training data to {filepath}")
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found at {os.path.join(data_dir, 'train.csv')}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        features = self.features[idx]
        if self.transform:
            features = self.transform(features)
        
        return features, self.labels[idx]

class TestDataset(Dataset):
    def __init__(self, data_dir: str, transform: v2.Compose = None):
        self.transform = transform

        filepath = os.path.join(data_dir, "test.pt")
        if os.path.exists(filepath):
            data = torch.load(filepath)
            self.features = data["features"]
        else:
            try:
                df = pd.read_csv(os.path.join(data_dir, "test.csv"))
                self.features = torch.tensor(df.values.astype("float32")).reshape(-1, 1, 28, 28)
                torch.save({"features": self.features}, filepath)
                print(f"Saved processed test data to {filepath}")
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found at {os.path.join(data_dir, 'test.csv')}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        features = self.features[idx]
        if self.transform:
            features = self.transform(features)

        return features
