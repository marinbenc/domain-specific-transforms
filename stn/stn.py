# based on https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
# TODO based on STN tutorial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import segmentation_models_pytorch as smp

class STN(nn.Module):  
  def __init__(self, loc_net):
      super(STN, self).__init__()

      # Spatial transformer localization-network
      self.loc_net = loc_net

      # Regressor for the 3 * 2 affine matrix
      self.loc_head = nn.Sequential(
        nn.Linear(512 * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 3 * 2)
      )

      # Initialize the weights/bias with identity transformation
      self.loc_head[-1].weight.data.zero_()
      self.loc_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

  # Spatial transformer network forward function
  def stn(self, x):
      xs = self.loc_net(x)[-1]
      xs = xs.view(-1, 512 * 4 * 4)
      theta = self.loc_head(xs)
      theta = theta.view(-1, 2, 3)

      grid = F.affine_grid(theta, x.size())
      x = F.grid_sample(x, grid)

      return x

  def forward(self, x):
      x = self.stn(x)
      return x