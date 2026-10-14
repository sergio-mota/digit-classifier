import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


logger = logging.getLogger(__name__)

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
                logger.info(f"Saved processed training data to {filepath}")
            except FileNotFoundError:
                error_message = f"CSV file not found at {os.path.join(data_dir, 'train.csv')}"
                logger.error(error_message)
                raise FileNotFoundError(error_message)

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
                logger.info(f"Saved processed test data to {filepath}")
            except FileNotFoundError:
                error_message = f"CSV file not found at {os.path.join(data_dir, 'teste.csv')}"
                logger.error(error_message)
                raise FileNotFoundError(error_message)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        features = self.features[idx]
        if self.transform:
            features = self.transform(features)

        return features

def get_data_loaders(data_dir : str, splits: list[float], batch_size: int, num_workers: int):
    logger.debug("Data configuration:")
    logger.debug(f"Data directory: {data_dir}")
    logger.debug(f"Batch size: {batch_size}")
    logger.debug(f"Number of workers: {num_workers}")
    logger.debug(f"Dataset splits: {splits}")

    transform = v2.Compose([
        v2.Normalize((0.5,), (0.5,))
    ])

    full_dataset = TrainDataset(data_dir, transform=transform)

    valid_size = int(len(full_dataset) * splits[0])
    train_size = len(full_dataset) - valid_size
    if len(splits) == 2:
        test_size = int(len(full_dataset) * splits[1])
        train_size -= test_size
        print(f"Full dataset size: {len(full_dataset)}")
        print(f"Train size: {train_size}, Validation size: {valid_size}, Test size: {test_size}")
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size])
    else:
        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
        test_dataset = TestDataset(data_dir, transform=transform)

    logger.debug(f"Train dataset size: {len(train_dataset)}")
    logger.debug(f"Validation dataset size: {len(valid_dataset)}")
    logger.debug(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "validation": valid_loader, "test": test_loader}