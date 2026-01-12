from pathlib import Path

import typer
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch


class corruptedMNIST(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)
        # split-aware attributes
        self.train_images = None
        self.train_targets = None
        self.test_images = None
        self.test_targets = None

        # compatibility convenience attributes (keeps legacy behaviour)
        self.images = None
        self.targets = None
        # Try to load data on initialization so the dataset is ready to use.
        try:
            self._load_data()
        except Exception:
            # Do not raise here; keep lazy behavior for callers that only
            # want to create the dataset object without immediate I/O.
            pass

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.targets) if self.targets is not None else 0

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if self.images is None or self.targets is None:
            self._load_data()
        if self.images is None or self.targets is None:
            raise RuntimeError("Processed data not found. Run preprocess first.")

        return self.images[index], self.targets[index]

    def _load_data(self) -> None:
        """Attempt to load processed tensors; if missing, try to preprocess raw data.

        This centralizes load logic so it's used both at init and lazily in
        `__getitem__`.
        """
        processed_dir = self.data_path / "processed" / "corruptmnist_v1"
        if not processed_dir.exists():
            processed_dir = self.data_path / "corruptmnist_v1"

        train_images_path = processed_dir / "train_images.pt"
        train_targets_path = processed_dir / "train_targets.pt"
        test_images_path = processed_dir / "test_images.pt"
        test_targets_path = processed_dir / "test_targets.pt"

        # Load whatever processed tensors are available
        if train_images_path.exists():
            self.train_images = torch.load(train_images_path)
        if train_targets_path.exists():
            self.train_targets = torch.load(train_targets_path)
        if test_images_path.exists():
            self.test_images = torch.load(test_images_path)
        if test_targets_path.exists():
            self.test_targets = torch.load(test_targets_path)

        # If none of the processed files exist, try preprocessing raw data
        if not any(p.exists() for p in [train_images_path, train_targets_path, test_images_path, test_targets_path]):
            raw_dir = self.data_path / "raw" / "corruptmnist_v1"
            if not raw_dir.exists():
                raw_dir = self.data_path / "corruptmnist_v1"

            if raw_dir.exists():
                output_folder = self.data_path / "processed"
                self.preprocess(output_folder)
                processed_dir = output_folder / "corruptmnist_v1"
                train_images_path = processed_dir / "train_images.pt"
                train_targets_path = processed_dir / "train_targets.pt"
                test_images_path = processed_dir / "test_images.pt"
                test_targets_path = processed_dir / "test_targets.pt"
                if train_images_path.exists():
                    self.train_images = torch.load(train_images_path)
                if train_targets_path.exists():
                    self.train_targets = torch.load(train_targets_path)
                if test_images_path.exists():
                    self.test_images = torch.load(test_images_path)
                if test_targets_path.exists():
                    self.test_targets = torch.load(test_targets_path)

        # Keep legacy `images`/`targets` attributes for compatibility:
        # - `images` defaults to training images (if available)
        # - `targets` defaults to test targets (if available)
        if self.train_images is not None:
            self.images = self.train_images
        elif self.test_images is not None:
            self.images = self.test_images

        if self.test_targets is not None:
            self.targets = self.test_targets
        elif self.train_targets is not None:
            self.targets = self.train_targets

        # If neither were found, leave them as None

    def preprocess(self, output_folder: Path) -> None:
        """Return train and test dataloaders for corrupt MNIST.
        This concatenates per-shard tensors, ensures a channel dimension,
        and returns DataLoader objects that yield batches of shape
        `(batch, 1, H, W)` and targets with `torch.long` dtype.
        """
        raw_dir = self.data_path / "raw" / "corruptmnist_v1"
        if not raw_dir.exists():
            # allow passing directly the raw folder as data_path
            raw_dir = self.data_path / "corruptmnist_v1"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")

        train_images = [torch.load(raw_dir / f"train_images_{i}.pt") for i in range(6)]
        train_targets = [torch.load(raw_dir / f"train_target_{i}.pt") for i in range(6)]
        train_images = torch.cat(train_images, dim=0)
        train_targets = torch.cat(train_targets, dim=0)

        # add channel dim and ensure types
        train_images = train_images.unsqueeze(1).float()
        train_targets = train_targets.long()

        test_images = torch.load(raw_dir / "test_images.pt")
        test_targets = torch.load(raw_dir / "test_target.pt")
        test_images = test_images.unsqueeze(1).float()
        test_targets = test_targets.long()

        # ensure output folder exists and save processed tensors
        processed_dir = Path(output_folder) / "corruptmnist_v1"
        processed_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train_images, processed_dir / "train_images.pt")
        torch.save(train_targets, processed_dir / "train_targets.pt")
        torch.save(test_images, processed_dir / "test_images.pt")
        torch.save(test_targets, processed_dir / "test_targets.pt")

        print(f"Saved processed data to: {processed_dir}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = corruptedMNIST(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
