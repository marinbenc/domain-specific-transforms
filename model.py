import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import utils

class TransformedSegmentation(nn.Module):
  """
  Transformed Segmentation Network

  Attributes:
    stn: The spatial transformer network.
    seg: The segmentation network.
    output_theta: if `True`, the model output will be `(y, theta)`, otherwise it will be `y`
  """
  def __init__(self, stn, seg):
      super(TransformedSegmentation, self).__init__()

      stn.output_theta = True

      # Spatial transformer localization-network
      self.stn = stn
      self.seg = seg
      self._iters = 0
      self.output_theta = False

  def forward(self, x):
    # transform image
    x_t, theta_out = self.stn(x)
    # segment transformed image
    y_t = self.seg(x_t)
    
    # make theta square
    row = torch.tensor([0, 0, 1], dtype=theta_out.dtype, device=theta_out.device).expand(theta_out.shape[0], 1, 3)
    theta = torch.cat([theta_out, row], dim=1)
    theta_inv = torch.inverse(theta)[:, :2, :]
    grid = F.affine_grid(theta_inv, y_t.size())
    y = F.grid_sample(y_t, grid)

    #utils.show_torch(imgs=[x[0] + 0.5, x_t[0] + 0.5, y_t[0], y[0]])

    # self._iters += 1

    # if self._iters % 100 == 0:
    #   utils.show_torch(imgs=[x[0] + 0.5, x_t[0] + 0.5, y_t[0], y[0]])
    
    if self.output_theta:
      return y, theta_out
    else:
      return y
