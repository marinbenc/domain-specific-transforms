import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from segment_anything import sam_model_registry, SamPredictor

class SAMSegmenter(nn.Module):
  """
  Attributes:
    padding: The padding of the bounding box in the original image space.

  Methods:
    forward(x):
      Forward pass of the model. The input `x` is a dictionary with the following keys:
        - 'img_stn_th': STN-transformed image (with thresholding), with shape (batch_size, 1, H, W)
        - 'theta_inv': the inverse of the 3 * 2 affine matrix output by the STN, with shape (batch_size, 3, 2)
      The output is a segmentation mask with shape (batch_size, 1, H, W), calculated with GrabCut.
  """
  def __init__(self, padding):
    super(GrabCutSegmenter, self).__init__()
    self.padding = padding
    self.sam = sam_model_registry['vit_h'](checkpoint="sam_checkpoints/sam_vit_h_4b8939.pth")
    self.sam.to('cuda')
    self.predictor = SamPredictor(self.sam)

  def _is_valid_bbox(self, bbox, img_shape):
    return (
      bbox[0] >= 0 and 
      bbox[1] >= 0 and 
      bbox[2] < img_shape[1] and
      bbox[2] > 0 and
      bbox[3] < img_shape[0] and
      bbox[3] > 0)

  def forward(self, x):
    img, theta_inv = x['img_th_stn'].clone(), x['theta_inv']
    predictor.set_image(img)
    # center point of the image
    
    center_label = 
    masks, _, _ = predictor.predict(<input_prompts>)
    # transform segmentation to original image space
    grid = F.affine_grid(theta_inv, img.size(), align_corners=True)
    img = F.grid_sample(img, grid, align_corners=True)
    return img

