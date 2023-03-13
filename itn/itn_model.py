# STN module is based on https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import utils

# TODO: Rename to threshold model or something like that
class ITN(nn.Module):
  """
  Image Transformer Network

  Attributes:
    loc_net: the localization network

    stn_transform: if `True`, the model will use STN. Useful for pre-training the thresholding.
      Note: if `stn_transform` is `False`, `theta`, `img_stn` and `img_th_stn` will be `None` 
      in the output dictionary.
    
    segmentation_model: the segmentation model or None if no segmentation is needed. If `None`,
      `seg` will be `None` in the output dictionary. 
      
      The model must be a PyTorch model with a forward method that takes an input dictionary 
      with keys:
        - 'img_th_stn': STN-transformed image (with thresholding)
        - 'theta_inv': the inverse of the 3 * 2 affine matrix output by the STN with 
                       shape (batch_size, 3, 2)
      and returns a segmentation mask with shape (batch_size, 1, H, W). The segmentation mask 
      must be in the original image space (i.e. not transformed by the STN).

      Built in segmentation models:
        - GrabCutSegmentationModel in weak_annotation.py
        - TODO

  Returns: 
    A dictionary with keys:
      - 'img_th': thresholded image
      - 'threshold': the threshold with shape (batch_size, 2)
      - 'img_stn: STN-transformed image (without thresholding)
      - 'theta': the 3 * 2 affine matrix output by the STN with shape (batch_size, 3, 2)
      - 'img_th_stn': STN-transformed image (with thresholding)
      - 'seg': segmentation mask obtained with GrabCut
  """
  def __init__(self, loc_net, threshold=True, stn_transform=True, segmentation_model=None):
      super(ITN, self).__init__()

      self.threshold = threshold
      self.stn_transform = stn_transform
      self.segmentation_model = segmentation_model

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

  def transform(self, x, theta, size):
    grid = F.affine_grid(theta, size, align_corners=False)
    x = F.grid_sample(x, grid)
    return x

  def smooth_threshold(self, x, low, high):
    slope = 50
    th_low = 1 / (1 + torch.exp(slope * (low - x)))
    th_high = 1 / (1 + torch.exp(-slope * (high - x)))
    return th_low + th_high - 1

  def forward(self, x):
    xs = self.loc_net(x)[-1]
    xs = self.avg_pool(xs)
    xs = xs.view(-1, 512 * 4 * 4)

    # Threshold the image
    if self.threshold:
      threshold = self.thresh_head(xs)
      th_low = threshold[:, 0].view(-1, 1, 1, 1)
      th_high = threshold[:, 1].view(-1, 1, 1, 1)
      mask = self.smooth_threshold(x, th_low, th_high)
      x_th = x * mask
    else:
      x_th = None
      threshold = None
    
    # Spatial transformer

    if self.stn_transform:
      theta = self.stn_head(xs)
      theta = theta.view(-1, 2, 3)

      grid = F.affine_grid(theta, x.size(), align_corners=False)
      x = F.grid_sample(x, grid)
      mask = F.grid_sample(mask, grid)
      if x_th is not None:
        x_th_stn = F.grid_sample(x_th, grid)
      else:
        x_th_stn = None
    
    #print(theta)
    #plt.imshow(x_th_stn[0, 0].detach().cpu().numpy())
    #plt.show()

    row = torch.tensor([0, 0, 1], dtype=theta.dtype, device=theta.device).expand(theta.shape[0], 1, 3)
    theta_sq = torch.cat([theta, row], dim=1)
    theta_inv = torch.inverse(theta_sq)[:, :2, :]
    grid = F.affine_grid(theta_inv, x.size())
    mask = self.transform(mask, theta_inv, x.size())

    if self.segmentation_model is not None:
      seg = self.segmentation_model({'img_th_stn': x_th_stn, 'theta_inv': theta_inv})
    else:
      seg = None

    output = {
      'img_th': x_th,
      'threshold': threshold,
      'img_stn': x,
      'theta': theta,
      'theta_inv': theta_inv,
      'img_th_stn': x_th_stn,
      'seg': seg,
    }

    return output