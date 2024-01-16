# based on https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
# TODO based on STN tutorial

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import matplotlib.pyplot as plt

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

      self.fc = nn.Linear(512 * 4 * 4, 32)
      self.translation = nn.Linear(32, 2)
      self.rotation = nn.Linear(32, 1)
      self.scaling = nn.Linear(32, 2)
      self.shearing = nn.Linear(32, 1)

      self.translation.weight.data.zero_()
      self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
      self.rotation.weight.data.zero_()
      self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
      self.scaling.weight.data.zero_()
      self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
      self.shearing.weight.data.zero_()
      self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))


      # TODO: Experiment with using fully conv network
      # # Regressor for the 3 * 2 affine matrix
      # self.loc_head = nn.Sequential(
      #   nn.Conv2d(512, 1, (7, 6))
      # )

      # # Initialize the weights/bias with identity transformation
      # self.loc_head[-1].bias.data.zero_()
      # nn.init.dirac_(self.loc_head[-1].weight)

  def stn_transform(x, theta):
    grid = F.affine_grid(theta, x.size(), align_corners=False).to(x.device)
    x = F.grid_sample(x, grid)
    return x

  def affine_matrix(self, x):
    b = x.size(0)

    # trans = self.translation(x)
    trans = torch.tanh(self.translation(x)) * 0.1
    translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
    translation_matrix[:, 0, 0] = 1.0
    translation_matrix[:, 1, 1] = 1.0
    translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
    translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
    translation_matrix[:, 2, 2] = 1.0

    # rot = self.rotation(x)
    rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
    rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
    rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
    rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
    rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
    rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
    rotation_matrix[:, 2, 2] = 1.0

    # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
    # scale = self.scaling(x)
    scale = torch.tanh(self.scaling(x)) * 0.2
    scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
    # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
    # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
    scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
    scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
    scaling_matrix[:, 2, 2] = 1.0

    # shear = self.shearing(x)
    shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
    shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
    shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
    shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
    shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
    shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
    shearing_matrix[:, 2, 2] = 1.0

    # Affine transform
    matrix = torch.bmm(shearing_matrix, scaling_matrix)
    matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
    matrix = torch.bmm(matrix, rotation_matrix)
    matrix = torch.bmm(matrix, translation_matrix)

    # matrix = torch.bmm(translation_matrix, rotation_matrix)
    # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
    # matrix = torch.bmm(matrix, scaling_matrix)
    # matrix = torch.bmm(matrix, shearing_matrix)

    # No-shear transform
    # matrix = torch.bmm(scaling_matrix, rotation_matrix)
    # matrix = torch.bmm(matrix, translation_matrix)

    # Rigid-body transform
    # matrix = torch.bmm(rotation_matrix, translation_matrix)

    matrix = torch.bmm(scaling_matrix, translation_matrix)

    return matrix[:, 0:2, :]

  # Spatial transformer network forward function
  def stn(self, x):
      x_original = x.detach().clone()
      features = self.loc_net.encoder(x)

      # Used during training as a way to achieve deep supervision
      decoder_output = self.loc_net.decoder(*features)
      y_mask = self.loc_net.segmentation_head(decoder_output)

      xs = features[-1]
      xs = self.avg_pool(xs)
      xs = xs.view(-1, 512 * 4 * 4)
      xs = F.relu(self.fc(xs))
      theta = self.affine_matrix(xs)

      # Use the following line to test how sensitive SEG is to STN
      # by uncommenting running the test with --transformed-images and stn-to-seg type
      #theta = torch.tensor([0.95, 0, 0, 0, 1.05, 0], dtype=torch.float).cuda()

      # without augmentation for SEG, even tiny changes in theta (like above) can cause
      # the segmentation DSC to drop significantly
      
      #torch.set_printoptions(precision=3, sci_mode=False)
      #print(theta[0])

      grid = F.affine_grid(theta, x.size(), align_corners=False).to(x.device)
      x = F.grid_sample(x, grid)

      y_mask_t = F.grid_sample(y_mask, grid)

      #utils.show_torch([x_original[0], x[0]])

      return y_mask, y_mask_t, x, theta

  def forward(self, x):
      viz = False
      if viz:
        plt.imshow(x[0].detach().cpu().numpy().transpose(1, 2, 0) + 0.5)
        plt.show()
      
      y_mask, y_mask_t, x, theta = self.stn(x)

      if viz:
        plt.imshow(x[0].detach().cpu().numpy().transpose(1, 2, 0) + 0.5)
        plt.show()
      
      if self.output_theta:
        return (y_mask, y_mask_t, x, theta)
      else:
        return (y_mask, y_mask_t, x)
