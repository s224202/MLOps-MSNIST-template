import torch
from unittest import mock

from mlops_mnist_from_template import train as train_module


class FakeDataset:
	def __init__(self, path=None):
		self._len = 4

	def __len__(self):
		return self._len

	def __getitem__(self, idx):
		img = torch.zeros(1, 1, 28, 28)
		target = torch.tensor(0, dtype=torch.long)
		return img, target


class FakeModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = torch.nn.Linear(28 * 28, 10)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.lin(x)


def test_train_minimal(tmp_path):
	fake_save = mock.MagicMock()
	with mock.patch("mlops_mnist_from_template.train.corruptedMNIST", return_value=FakeDataset()), \
		mock.patch("mlops_mnist_from_template.train.Model", FakeModel), \
		mock.patch("mlops_mnist_from_template.train.torch.save", fake_save):
		model = train_module.train(data_path=str(tmp_path), batch_size=2, epochs=1, device="cpu")

	assert isinstance(model, FakeModel)
	assert fake_save.called

