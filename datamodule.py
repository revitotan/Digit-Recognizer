from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import lightning as L

class DataModule(L.LightningDataModule):
  def __init__(self, data_dir='./mnist', batch_size = 64, num_workers=0, train_transform=None, test_transform=None):
    super().__init__()

    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.train_transform = train_transform
    self.test_transform = test_transform

  def prepare_data(self):
    datasets.MNIST(self.data_dir, train=True, download=True)
    datasets.MNIST(self.data_dir, train=False, download=True)

  def setup(self, stage=None):
    self.test_data = datasets.MNIST(
      root=self.data_dir, 
      train=False,
      download=False, 
      transform=self.test_transform
    )
    mnist_full = datasets.MNIST(
      root=self.data_dir, 
      train=True, 
      download=False,
      transform=self.train_transform,
    )
    self.train_data, self.val_data = random_split(mnist_full, [55000, 5000])

  def train_dataloader(self):
    return DataLoader(
      dataset=self.train_data, 
      batch_size=self.batch_size, 
      shuffle=True, 
      drop_last=True,
      num_workers=self.num_workers, 
    )

  def val_dataloader(self):
    return DataLoader(
      dataset=self.val_data, 
      batch_size=self.batch_size, 
      shuffle=False,
      drop_last=False,
      num_workers=self.num_workers,
    )

  def test_dataloader(self):
    return DataLoader(
      dataset=self.test_data, 
      batch_size=self.batch_size, 
      shuffle=False,
      drop_last=False,
      num_workers=self.num_workers,
    )