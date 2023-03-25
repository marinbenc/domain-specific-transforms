# STN module is based on https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

import utils
# TODO: Remove
from segmenters.grabcut_segmenter import GrabCutSegmenter
from segmenters.cnn_segmenter import CNNSegmenter

def pre_cut_classification_loss(output, target):
  loss = F.binary_cross_entropy_with_logits(output['is_empty'], target.unsqueeze(-1))
  return loss

def pre_cut_loss(output, target, threshold_loss_weight=1.0):
  output_theta, target_theta = output['theta'], target['theta']
  output_thresh, target_thresh = output['threshold'], target['threshold']
  th_loss = F.mse_loss(output_thresh, target_thresh)
  stn_loss = F.mse_loss(output_tensor, target_tensor)
  return stn_loss + threshold_loss_weight * th_loss

def get_unet(dataset, device, checkpoint=None):
  unet = smp.Unet('resnet18', in_channels=dataset.in_channels, classes=1, 
                  activation='sigmoid', decoder_use_batchnorm=True)
                  #encoder_depth=3, decoder_channels=(128, 64, 16))
  unet = unet.to(device)
  if checkpoint is not None:
    saved_unet = torch.load(checkpoint)
    unet.load_state_dict(saved_unet['model'])
  return unet

def _get_unet_loc_net(dataset, device, checkpoint=None):
  unet = get_unet(dataset, device, checkpoint)
  return unet.encoder

def get_model(segmentation_method, dataset, pretrained_unet=None, pretrained_precut=None, device='cuda', **segmenter_kwargs):
  """
  Returns a PreCut model with the specified segmentation method.

  Args:
    segmentation_method: the segmentation method to use. One of 'grabcut', 'cnn' or 'none'. If 'cnn' the model will use U-Net.
    dataset: the dataset to use (used to get the padding and input channel number)
    pretrained_unet: the checkpoint path to a pre-trained U-Net or None. The U-Net will be used to pre-train loc_net
      as well as the segmentation model if `segmentation_method` is 'cnn'.
    device: the device to use
  """
  loc_net = _get_unet_loc_net(dataset, device, pretrained_unet)
  
  if segmentation_method == 'grabcut':
    segmentation_model = GrabCutSegmenter(padding=dataset.padding, **segmenter_kwargs)
  elif segmentation_method == 'cnn':
    unet = get_unet(dataset, device, pretrained_unet)
    segmentation_model = CNNSegmenter(padding=dataset.padding, segmentation_model=unet, **segmenter_kwargs)
  elif segmentation_method == 'none':
    segmentation_model = None
  else:
    raise ValueError(f'Invalid segmentation method: {segmentation_method}')
  
  # TODO: Rename dataset.width to dataset.size or similar
  model = PreCut(loc_net=loc_net, segmentation_model=segmentation_model, input_size=dataset.width)
  if pretrained_precut is not None:
    print('Pretraining PreCut...')
    saved_model = torch.load(pretrained_precut)
    saved_model = saved_model['model']
    model.loc_net.load_state_dict({k.replace('loc_net.', ''): v for k, v in saved_model.items() if k.startswith('loc_net')})
    model.thresh_head.load_state_dict({k.replace('thresh_head.', ''): v for k, v in saved_model.items() if k.startswith('thresh_head')})
    model.stn_head.load_state_dict({k.replace('stn_head.', ''): v for k, v in saved_model.items() if k.startswith('stn_head')})

    # Freeze PreCut parameters if fine-tuning a segmentation model
    for p in list(model.loc_net.parameters()) + list(model.thresh_head.parameters()) + list(model.stn_head.parameters()):
      p.requires_grad = False
  model.to(device)
  return model

class PreCut(nn.Module):
  """
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

      Built in segmentation models (see segmenters/):
        - GrabCutSegmenter: uses OpenCV's GrabCut algorithm
        - CNNSegmenter: uses a U-Net to segment the image

    **segmenter_kwargs: keyword arguments to pass to the segmentation model

  Returns: 
    A dictionary with keys:
      - 'img_th': thresholded image
      - 'threshold': the threshold with shape (batch_size, 2)
      - 'img_stn: STN-transformed image (without thresholding)
      - 'theta': the 3 * 2 affine matrix output by the STN with shape (batch_size, 3, 2)
      - 'img_th_stn': STN-transformed image (with thresholding)
      - 'seg': segmentation mask obtained with GrabCut
  """
  def __init__(self, loc_net, input_size=512, threshold=True, stn_transform=True, segmentation_model=None):
      super(PreCut, self).__init__()

      self.threshold = threshold
      self.stn_transform = stn_transform
      self.segmentation_model = segmentation_model
      self.input_size = input_size

      # TODO: Move this to a parameter in init, rename input_size to encoder_output_depth or something
      self.input_size = loc_net._out_channels[loc_net._depth]

      # Spatial transformer localization-network
      self.loc_net = loc_net
      self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

      # Regressor for the threshold
      self.thresh_head = nn.Sequential(
        nn.Linear(self.input_size * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 2)
      )

      # Initialize to untresholded image
      self.thresh_head[-1].weight.data.zero_()
      self.thresh_head[-1].bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))

      # Regressor for the 3 * 2 affine matrix
      self.stn_head = nn.Sequential(
        nn.Linear(self.input_size * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 3 * 2)
      )

      # Initialize to identity transformation
      self.stn_head[-1].weight.data.zero_()
      self.stn_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

      # Regressor for the empty slice classification
      self.is_empty_head = nn.Sequential(
        nn.Linear(self.input_size * 4 * 4, 128),
        nn.ReLU(True),
        nn.Linear(128, 1)
      )

  def transform(self, x, theta, size):
    grid = F.affine_grid(theta, size, align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)
    return x

  def smooth_threshold(self, x, low, high):
    slope = 50
    th_low = 1 / (1 + torch.exp(slope * (low - x)))
    th_high = 1 / (1 + torch.exp(-slope * (high - x)))
    return th_low + th_high - 1

  def forward(self, x):
    xs = self.loc_net(x)[-1]
    xs = self.avg_pool(xs)
    xs = xs.view(-1, self.input_size * 4 * 4)
    is_empty  = self.is_empty_head(xs)

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

      grid = F.affine_grid(theta, x.size(), align_corners=True)
      x = F.grid_sample(x, grid, align_corners=True)
      mask = F.grid_sample(mask, grid, align_corners=True)
      if x_th is not None:
        x_th_stn = F.grid_sample(x_th, grid, align_corners=True)
      else:
        x_th_stn = None
    
    #print(theta)
    # plt.imshow(x_th_stn[0, 0].detach().cpu().numpy())
    # plt.show()

    row = torch.tensor([0, 0, 1], dtype=theta.dtype, device=theta.device).expand(theta.shape[0], 1, 3)
    theta_sq = torch.cat([theta, row], dim=1)
    try:
      theta_inv = torch.inverse(theta_sq)[:, :2, :]
    except:
      theta_inv = torch.zeros_like(theta_sq)[:, :2, :]
    grid = F.affine_grid(theta_inv, x.size())
    mask = self.transform(mask, theta_inv, x.size())

    if self.segmentation_model is not None:
      seg = self.segmentation_model({'img_th_stn': x_th_stn, 'theta_inv': theta_inv})
      #is_empty_class = torch.sigmoid(is_empty) > 0.5
      #seg[is_empty_class] = torch.zeros_like(seg[is_empty_class])
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
      'is_empty': is_empty
    }

    return output