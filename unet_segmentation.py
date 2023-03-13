import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

import segmentation_models_pytorch as smp

class UNetSegmentation(nn.Module):
  """
  U-Net Segmentation Model

  Used as the segmentation module of the model.

  Attributes:
    padding: The padding of the bounding box in the original image space.
    sigmoid_activation: Whether to use a sigmoid activation function in the last layer of the U-Net.

  Methods:
    forward(x):
      Forward pass of the model. The input `x` is a dictionary with the following keys:
        - 'img_stn_th': STN-transformed image (with thresholding), with shape (batch_size, 1, H, W)
        - 'theta_inv': the inverse of the 3 * 2 affine matrix output by the STN, with shape (batch_size, 3, 2)
      The output is a segmentation mask with shape (batch_size, 1, H, W), calculated with GrabCut.
  """
  def __init__(self, padding, sigmoid_activation=False):
    super(UNetSegmentation, self).__init__()
    self.unet = smp.Unet('resnet18', in_channels=dataset.in_channels, classes=1, 
                         activation='sigmoid' if sigmoid_activation else None, decoder_use_batchnorm=True)
    self.padding = padding

    self.parameters = list(self.unet.parameters())

  def forward(self, x):
    img, theta_inv = x['img_th_stn'], x['theta_inv']
    mask = self.unet(img)
    grid = F.affine_grid(theta_inv, mask.shape)
    mask = F.grid_sample(mask, grid)
    return mask