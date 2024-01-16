import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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
      self.output_stn_mask = True

      # Spatial transformer localization-network
      self.stn = stn
      self.seg = seg
      self._iters = 0
      self.output_theta = False

  def forward(self, x):
    # transform image
    y_inital, y_initial_t, x_t, theta_out = self.stn(x)

    # segment transformed image

    # plt.imshow(y_loc_net[0, 0].detach().cpu().numpy())
    # plt.title('y_loc_net')
    # plt.show()
    # plt.imshow(y_loc_net_t[0, 0].detach().cpu().numpy())
    # plt.title('y_loc_net_t')
    # plt.show()

    skip_stn = False

    # two channels, y_loc_net_t is the mask and x_t is the transformed image
    if skip_stn:
      seg_x = x
    else:
      # First channel: x_t
      # Second channel: y_initial_t
      seg_x = torch.cat([x_t, y_initial_t], dim=1)
    y_t = self.seg(seg_x)

    # plt.imshow(y_t[0, 0].detach().cpu().numpy())
    # plt.title('y_t')
    # plt.show()
    
    # make theta square
    if not skip_stn:
      row = torch.tensor([0, 0, 1], dtype=theta_out.dtype, device=theta_out.device).expand(theta_out.shape[0], 1, 3)
      theta = torch.cat([theta_out, row], dim=1)
      theta_inv = torch.inverse(theta)[:, :2, :]
      grid = F.affine_grid(theta_inv, y_t.size()).to(x.device)
      y = F.grid_sample(y_t, grid)
    else:
      y = y_t

    #utils.show_torch(imgs=[x[0] + 0.5, x_t[0] + 0.5, y_t[0], y[0]])

    # self._iters += 1

    # if self._iters % 100 == 0:
    #   utils.show_torch(imgs=[x[0] + 0.5, x_t[0] + 0.5, y_t[0], y[0]])

    outputs = [y]
    if self.output_stn_mask:
      outputs = [y_inital] + outputs
    if self.output_theta:
      outputs = outputs + [theta_out]

    if len(outputs) == 1:
      return outputs[0]
    else:
      return tuple(outputs)
