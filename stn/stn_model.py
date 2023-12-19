# based on https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
# TODO based on STN tutorial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import segmentation_models_pytorch as smp

import utils

class STN(nn.Module):
  """
  Spatial Transformer Network

  Attributes:
    loc_net: the localization network
    output_theta: if `True`, the model output will be `(y_mask, y, theta)`, otherwise it will be `(y_mask, y)`
  """
  def __init__(self, loc_net, output_theta = False):
      super(STN, self).__init__()

      self.output_theta = output_theta

      # Spatial transformer localization-network
      self.loc_net = loc_net
      self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

      # Regressor for the 3 * 2 affine matrix
      self.loc_head = nn.Sequential(
        nn.Linear(512 * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 3 * 2)
      )

      # Initialize the weights/bias with identity transformation
      self.loc_head[-1].weight.data.zero_()
      self.loc_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

      # TODO: Experiment with using fully conv network
      # # Regressor for the 3 * 2 affine matrix
      # self.loc_head = nn.Sequential(
      #   nn.Conv2d(512, 1, (7, 6))
      # )

      # # Initialize the weights/bias with identity transformation
      # self.loc_head[-1].bias.data.zero_()
      # nn.init.dirac_(self.loc_head[-1].weight)


  # Spatial transformer network forward function
  def stn(self, x):
      x_original = x.detach().clone()
      features = self.loc_net.encoder(x)

      decoder_output = self.loc_net.decoder(*features)
      y_mask = self.loc_net.segmentation_head(decoder_output)

      xs = features[-1]      
      xs = self.avg_pool(xs)
      xs = xs.view(-1, 512 * 4 * 4)
      theta = self.loc_head(xs)

      # Use the following line to test how sensitive SEG is to STN
      # by uncommenting running the test with --transformed-images and stn-to-seg type
      #theta = torch.tensor([0.95, 0, 0, 0, 1.05, 0], dtype=torch.float).cuda()

      # without augmentation for SEG, even tiny changes in theta (like above) can cause
      # the segmentation DSC to drop significantly
      
      theta = theta.view(-1, 2, 3)

      grid = F.affine_grid(theta, x.size(), align_corners=False)
      x = F.grid_sample(x, grid)

      y_mask_t = F.grid_sample(y_mask, grid)

      #utils.show_torch([x_original[0], x[0]])

      return y_mask, y_mask_t, x, theta

  def forward(self, x):
      y_mask, y_mask_t, x, theta = self.stn(x)
      if self.output_theta:
        return (y_mask, y_mask_t, x, theta)
      else:
        return (y_mask, y_mask_t, x)

class STN_CNN(nn.Module):
  """
  Spatial Transformer Network

  Attributes:
    loc_net: the localization network
    output_theta: if `True`, the model output will be `(y, theta)`, otherwise it will be `y`
  """
  def __init__(self, loc_net, output_theta = False):
      super(STN, self).__init__()

      self.output_theta = output_theta

      # Spatial transformer localization-network
      self.loc_net = loc_net
      self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

      # Regressor for the 3 * 2 affine matrix
      self.loc_head = nn.Sequential(
        
      )

      # Initialize the weights/bias with identity transformation
      self.loc_head[-1].weight.data.zero_()
      self.loc_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

      # TODO: Experiment with using fully conv network
      # # Regressor for the 3 * 2 affine matrix
      # self.loc_head = nn.Sequential(
      #   nn.Conv2d(512, 1, (7, 6))
      # )

      # # Initialize the weights/bias with identity transformation
      # self.loc_head[-1].bias.data.zero_()
      # nn.init.dirac_(self.loc_head[-1].weight)


  # Spatial transformer network forward function
  def stn(self, x):
      x_original = x.detach().clone()
      xs = self.loc_net(x)[-1]
      
      xs = self.avg_pool(xs)
      xs = xs.view(-1, 512 * 4 * 4)
      theta = self.loc_head(xs)

      # Use the following line to test how sensitive SEG is to STN
      # by uncommenting running the test with --transformed-images and stn-to-seg type
      #theta = torch.tensor([0.95, 0, 0, 0, 1.05, 0], dtype=torch.float).cuda()

      # without augmentation for SEG, even tiny changes in theta (like above) can cause
      # the segmentation DSC to drop significantly
      
      theta = theta.view(-1, 2, 3)

      grid = F.affine_grid(theta, x.size(), align_corners=False)
      x = F.grid_sample(x, grid)

      #utils.show_torch([x_original[0], x[0]])

      return x, theta

  def forward(self, x):
      x, theta = self.stn(x)
      if self.output_theta:
        return (x, theta)
      else:
        return x