from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mlops_mnist_from_template.model import Model
from mlops_mnist_from_template.data import corruptedMNIST


def train(data_path: str = "data", batch_size: int = 64, lr: float = 1e-3, epochs: int = 10, device: str | None = None):
    """Train a model on corrupt MNIST.

    Parameters
    - data_path: path to the project data folder (should contain `raw/corruptmnist_v1` or `processed/corruptmnist_v1`).
    - batch_size: minibatch size for training.
    - lr: learning rate for Adam optimizer.
    - epochs: number of epochs to train.
    - device: torch device string (e.g. 'cpu' or 'cuda'); if None, auto-selects.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = corruptedMNIST(Path(data_path))
    # Force dataset to load processed tensors so __len__ becomes correct.
    try:
        _ = dataset[0]
    except Exception as exc:
        raise RuntimeError("Failed to load processed data. Run the preprocessing step first.") from exc

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        epoch_loss = running_loss / (batch_count or 1)
        print(f"Epoch {epoch}/{epochs} — loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "./models/model.pth")
    return model


if __name__ == "__main__":
    train()
