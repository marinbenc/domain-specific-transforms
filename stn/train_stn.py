from stn import STN
from toy_dataset import ToyDataset

import torch
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import segmentation_models_pytorch as smp

device = 'cuda'

def train(model, optimizer, epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        fig, ax = plt.subplots()
        ax.imshow(data[0, 0].cpu().detach().numpy() * 0.25 + output[0,0].cpu().detach().numpy() * 0.75)
        rect = patches.Rectangle((40, 20), 20, 60, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

def main():
  train_loader = data.DataLoader(ToyDataset(), batch_size=8)
  test_loader = data.DataLoader(ToyDataset(), batch_size=8)

  loc_net = smp.Unet('resnet18', in_channels=1, classes=2, activation=None)
  loc_net = loc_net.encoder
  model = STN(loc_net=loc_net)
  model.to('cuda')

  optimizer = optim.SGD(model.parameters(), lr=0.01)

  for epoch in range(1, 200 + 1):
    train(model, optimizer, epoch, train_loader)






if __name__ == '__main__' :
  main()