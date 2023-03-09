import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

import utils

# TODO: Rename to threshold model or something like that
class ITN(nn.Module):
  """
  Image Transformer Network

  Attributes:
    loc_net: the localization network
    output_theta: if `True`, the model output contain theta in the form of (N, 2) tensor 
                  where N is the batch size and the 2 values are the low and high thresholds
    output_img: if `True`, the model output contain the thresholded image. 
                If both `output_theta` and `output_img` are `True`, the output will be (img, theta).
    segmentation_mode: if `True`, the model will use a sigmoid activation function on the thresholded image.
  """
  def __init__(self, loc_net, output_theta=False, output_img=True, segmentation_mode=False):
      super(ITN, self).__init__()

      self.output_theta = output_theta
      self.output_img = output_img
      self.segmentation_mode = segmentation_mode

      # Spatial transformer localization-network
      self.loc_net = loc_net
      self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

      # Regressor for the 3 * 2 affine matrix
      self.loc_head = nn.Sequential(
        nn.Linear(512 * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 2)
      )

      # Initialize the weights/bias with identity transformation
      self.loc_head[-1].weight.data.zero_()
      self.loc_head[-1].bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))

      # TODO: Experiment with using fully conv network
      # # Regressor for the 3 * 2 affine matrix
      # self.loc_head = nn.Sequential(
      #   nn.Conv2d(512, 1, (7, 6))
      # )

      # # Initialize the weights/bias with identity transformation
      # self.loc_head[-1].bias.data.zero_()
      # nn.init.dirac_(self.loc_head[-1].weight)

  def smooth_threshold(self, x, low, high):
    slope = 50
    th_low = 1 / (1 + torch.exp(slope * (low - x)))
    th_high = 1 / (1 + torch.exp(-slope * (high - x)))
    return th_low + th_high - 1

  def forward(self, x):
    assert self.output_img or self.output_theta, "Must output at least one of image or theta"
    
    x_original = x.detach().clone()
    xs = self.loc_net(x)[-1]
    
    xs = self.avg_pool(xs)
    xs = xs.view(-1, 512 * 4 * 4)
    theta = self.loc_head(xs)
      
    if self.output_img:
      th_low = theta[:, 0].view(-1, 1, 1, 1)
      th_high = theta[:, 1].view(-1, 1, 1, 1)
      if self.segmentation_mode:
        x = self.smooth_threshold(x, th_low, th_high)
      else:
        x = x * self.smooth_threshold(x, th_low, th_high)
    if self.output_img and self.output_theta:
      return x, theta
    elif self.output_img:
      return x
    elif self.output_theta:
      return theta