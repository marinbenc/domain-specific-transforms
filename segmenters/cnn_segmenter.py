import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()
    self.smooth = 1.0
    self.iters = 0

  @staticmethod
  def abs_exp_loss(y_pred, y_true, pow):
    return torch.abs((y_pred - y_true) ** pow).mean()

  def forward(self, y_pred, y_true):
    if isinstance(y_pred, dict):
      y_pred = y_pred['seg']
    if isinstance(y_true, dict):
      y_true = y_true['seg']

    dscs = torch.zeros(y_pred.shape[1])

    for i in range(y_pred.shape[1]):
      y_pred_ch = y_pred[:, i].contiguous().view(-1)
      y_true_ch = y_true[:, i].contiguous().view(-1)
      intersection = (y_pred_ch * y_true_ch).sum()
      dscs[i] = (2. * intersection + self.smooth) / (
          y_pred_ch.sum() + y_true_ch.sum() + self.smooth
      )

    return (1. - torch.mean(dscs))

class CNNSegmenter(nn.Module):
  """
  U-Net Segmentation Model

  Used as the segmentation module of the model.

  Attributes:
    segmentation_model: The segmentation model to use. A PyTorch model with a forward method 
      that takes an input image x with shape (batch_size, N, H, W) and returns a segmentation
      mask with shape (batch_size, 1, H, W).
    padding: The padding of the bounding box in the original image space.
    sigmoid_activation: Whether to use a sigmoid activation function in the last layer of the U-Net.

  Methods:
    forward(x):
      Forward pass of the model. The input `x` is a dictionary with the following keys:
        - 'img_stn_th': STN-transformed image (with thresholding), with shape (batch_size, 1, H, W)
        - 'theta_inv': the inverse of the 3 * 2 affine matrix output by the STN, with shape (batch_size, 3, 2)
      The output is a segmentation mask with shape (batch_size, 1, H, W), calculated with GrabCut.
  """
  def __init__(self, padding, segmentation_model):
    super(CNNSegmenter, self).__init__()
    self.segmentation_model = segmentation_model
    self.padding = padding

  def forward(self, x):
    img, theta_inv = x['img_th_stn'], x['theta_inv']
    mask = self.segmentation_model(img)
    grid = F.affine_grid(theta_inv, mask.shape, align_corners=True)
    mask = F.grid_sample(mask, grid, align_corners=True, mode='nearest')

    return mask
