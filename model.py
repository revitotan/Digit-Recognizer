import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
import lightning as L

class BaseModel(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()

    self.flatten = nn.Flatten()
    self.layers = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )

  def forward(self, x):
    x = self.flatten(x)
    return self.layers(x)
  


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5*5*16, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        logits = self.fc_layer(x)
        return logits
    

class ImprovedModel(nn.Module):
  def __init__(self, num_classes):
    super().__init__()

    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(6),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(5*5*16),
        nn.Linear(5*5*16, 120),
        nn.BatchNorm1d(120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.ReLU(),
        nn.Linear(84, num_classes)
    )

  def forward(self, x):
    x = self.conv_layers(x)
    return self.fc_layers(x)
  

class LightningModel(L.LightningModule):
  def __init__(self, model, lr):
    super().__init__()

    self.model = model
    self.lr = lr

    self.save_hyperparameters(ignore=["model"])

    self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)

  def forward(self, x):
    return self.model(x)

  def _shared_steps(self, batch):
    X, y = batch
    logits = self(X)
    loss = F.cross_entropy(logits, y)
    pred = torch.argmax(logits, dim=1)

    return loss, y, pred

  def training_step(self, batch, batch_idx):
    loss, y, pred = self._shared_steps(batch)
    self.log("train_loss", loss)
    self.train_acc(pred, y)
    self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)

    return loss

  def validation_step(self, batch, batch_idx):
    loss, y, pred = self._shared_steps(batch)
    self.log("val_loss", loss)
    self.val_acc(pred, y)
    self.log("val_acc", self.val_acc, prog_bar=True)

  def test_step(self, batch, batch_idx):
    loss, y, pred = self._shared_steps(batch)
    self.test_acc(pred, y)
    self.log("test_acc", self.test_acc)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)