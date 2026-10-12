import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
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

def get_data_loaders(data_dir : str, train_ratio: float, batch_size: int, num_workers: int):
    print("Data configuration:")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Train ratio: {train_ratio}")

    transform = v2.Compose([
        v2.Normalize((0.5,), (0.5,))
    ])

    full_dataset = TrainDataset(data_dir, transform=transform)

    train_size = int(len(full_dataset) * train_ratio)
    valid_size = len(full_dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    test_dataset = TestDataset(data_dir, transform=transform)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader