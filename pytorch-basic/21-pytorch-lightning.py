import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import os
from pytorch_lightning import Trainer

# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001



class LitNeuralNet(pl.LightningModule):
  def __init__(self, input_size, hidden_size, num_classes):
    super(LitNeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=learning_rate)



  def train_dataloader(self):
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=7, persistent_workers=True)
    return train_loader

  def training_step(self, batch, batch_idx):
    images, labels= batch
    images = images.reshape(-1, 28*28) # from [100, 1, 28, 28] -> [100, 784]
    outputs = self(images)
    loss = F.cross_entropy(outputs, labels)
    self.log("train_loss", loss)
    return loss



  def val_dataloader(self):
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)
    return val_loader

  def validation_step(self, batch, batch_idx):
    images, labels = batch
    images = images.reshape(-1, 28*28)
    outputs = self(images)
    loss = F.cross_entropy(outputs, labels)
    self.log("val_loss", loss)
    return loss
    

if __name__ == '__main__':
  trainer = Trainer(fast_dev_run=True, enable_progress_bar=True, log_every_n_steps=1)
  model = LitNeuralNet(input_size, hidden_size, num_classes)
  trainer.fit(model)