import torch
from torch.utils.data import Dataset

from mlops_mnist_from_template.data import corruptedMNIST


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = corruptedMNIST("data/raw")
    assert isinstance(dataset, Dataset)

def test_data_len():
    dataset = corruptedMNIST("data/processed")
    assert len(dataset) > 0
    assert len(dataset.images) == 30000
    assert len(dataset.targets) == 5000
    assert all(isinstance(dataset[i][0], torch.Tensor) and isinstance(dataset[i][1], torch.Tensor) for i in range(len(dataset)))
