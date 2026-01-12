import pytest
import torch
from mlops_mnist_from_template.model import Model
from mlops_mnist_from_template.data import corruptedMNIST
from mlops_mnist_from_template.train import train
import os.path

@pytest.mark.skipif(not os.path.exists("data/processed/corruptmnist_v1/train_images.pt"), reason="Processed data not found. Run preprocessing first.")
def test_io_shapes():
    dataset = corruptedMNIST("data/processed")
    sample_image, sample_label = dataset[0]
    assert sample_image.shape == (1, 28, 28)  # Assuming grayscale images of size 28x28
    assert isinstance(sample_label.item(), int)  # Assuming labels are integers
    model = Model()
    output = model(sample_image.unsqueeze(0))  # Add batch dimension
    assert output.shape == (1, 10)  # Assuming 10 classes for MNIST

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = Model()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)
