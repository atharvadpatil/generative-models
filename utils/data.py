"""Data loading utilities."""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(path: str) -> torch.Tensor:
    """Load CSV data as tensor."""
    df = pd.read_csv(path).astype("float32")
    return torch.from_numpy(df.values)


def create_dataloader(data: torch.Tensor, batch_size: int = 2048, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from tensor data."""
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle, drop_last=True)
