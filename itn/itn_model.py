import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

import utils

# TODO: Rename to threshold model or something like that
class ITN(nn.Module):
  #TODO: img_th should be only threshold, image_stn only stn, img_th_stn both
  """
  Image Transformer Network

  Attributes:
    loc_net: the localization network
  
  Returns: 
    A dictionary with keys:
      - 'img_th': thresholded image
      - 'threshold': the model output will be the threshold with shape (batch_size, 2)
      - 'img_stn: STN-transformed image
      - 'theta': the model output will be the 3 * 2 affine matrix with shape (batch_size, 3, 2)
      - 'seg': the model output will be the segmentation mask
  """
  def __init__(self, loc_net):
      super(ITN, self).__init__()

      # Spatial transformer localization-network
      self.loc_net = loc_net
      self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

      # Regressor for the threshold
      self.thresh_head = nn.Sequential(
        nn.Linear(512 * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 2)
      )

      # Initialize to untresholded image
      self.thresh_head[-1].weight.data.zero_()
      self.thresh_head[-1].bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))

      # Regressor for the 3 * 2 affine matrix
      self.stn_head = nn.Sequential(
        nn.Linear(512 * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 3 * 2)
      )

      # Initialize the weights/bias with identity transformation
      self.stn_head[-1].weight.data.zero_()
      self.stn_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

  def stn(self, x, xs):
    theta = self.stn_head(xs)
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid)
    return x, theta

  def smooth_threshold(self, x, low, high):
    slope = 50
    th_low = 1 / (1 + torch.exp(slope * (low - x)))
    th_high = 1 / (1 + torch.exp(-slope * (high - x)))
    return th_low + th_high - 1

  def forward(self, x):
    xs = self.loc_net(x)[-1]
    xs = self.avg_pool(xs)
    xs = xs.view(-1, 512 * 4 * 4)

    threshold = self.thresh_head(xs)

    # Threshold the image
    th_low = threshold[:, 0].view(-1, 1, 1, 1)
    th_high = threshold[:, 1].view(-1, 1, 1, 1)
    mask = self.smooth_threshold(x, th_low, th_high)
    x_th = x * mask

    # Spatial transformer

    # TODO: Try this 
    #xs = self.loc_net(x_th)[-1]
    #xs = self.avg_pool(xs)
    #xs = xs.view(-1, 512 * 4 * 4)

    theta = self.stn_head(xs)
    theta = theta.view(-1, 2, 3)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    x = F.grid_sample(x, grid)
    mask = F.grid_sample(mask, grid)
    x_th_stn = F.grid_sample(x_th, grid)

    row = torch.tensor([0, 0, 1], dtype=theta.dtype, device=theta.device).expand(theta.shape[0], 1, 3)
    theta_sq = torch.cat([theta, row], dim=1)
    theta_inv = torch.inverse(theta_sq)[:, :2, :]
    grid = F.affine_grid(theta_inv, x.size())
    mask = F.grid_sample(mask, grid)


    output = {
      'img_th': x_th,
      'threshold': threshold,
      'img_stn': x,
      'theta': theta,
      'img_th_stn': x_th_stn,
      'seg': mask
    }

    return output