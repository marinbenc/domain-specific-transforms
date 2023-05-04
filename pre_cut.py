# STN module is based on https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
from monai.networks.nets import FlexibleUNet

import utils
# TODO: Remove
from segmenters.grabcut_segmenter import GrabCutSegmenter
from segmenters.cnn_segmenter import CNNSegmenter
import data

def pre_cut_loss(output, target, threshold_loss_weight=1.0):
  output_theta, target_theta = output['theta'], target['theta']
  output_thresh, target_thresh = output['threshold'], target['threshold']
  th_loss = F.mse_loss(output_thresh, target_thresh)
  stn_loss = F.mse_loss(output_theta, target_theta)
  #print(f'Loss: {stn_loss.item():.4f} + {threshold_loss_weight:.2f} * {th_loss.item():.4f} = {stn_loss.item() + threshold_loss_weight * th_loss.item():.4f}')
  return stn_loss + threshold_loss_weight * th_loss

def get_unet_3d(dataset, device, checkpoint=None):
  unet = FlexibleUNet(
    spatial_dims=3,
    in_channels=dataset.in_channels,
    out_channels=1, # TODO: Multiple classes support
    backbone='efficientnet-b0',
    pretrained=True)
  unet.segmentation_head[-1] = nn.Sigmoid()
  unet = unet.to(device)
  if checkpoint is not None:
    saved_unet = torch.load(checkpoint)
    unet.load_state_dict(saved_unet['model'])
  return unet

def _get_unet_loc_net_3d(dataset, device, checkpoint=None):
  unet = get_unet_3d(dataset, device, checkpoint)
  return unet.encoder

def get_unet(dataset, device, checkpoint=None):
  is_3d = isinstance(dataset, data.scan_dataset.ScanDataset)
  if is_3d:
    return get_unet_3d(dataset, device, checkpoint)
  else:
    unet = smp.Unet('resnet18', in_channels=dataset.in_channels, classes=1, 
                    activation='sigmoid', decoder_use_batchnorm=True)
                    # TODO: Check if smaller model is better
                    #encoder_depth=3, decoder_channels=(128, 64, 16))
    unet = unet.to(device)
    if checkpoint is not None:
      saved_unet = torch.load(checkpoint)
      unet.load_state_dict(saved_unet['model'])
    return unet

def _get_unet_loc_net(dataset, device, checkpoint=None):
  unet = get_unet(dataset, device, checkpoint)
  return unet.encoder

def get_model(segmentation_method, dataset, pretrained_unet=None, pretrained_precut=None, pretraining=False, device='cuda', **segmenter_kwargs):
  """
  Returns a PreCut model with the specified segmentation method.

  Args:
    segmentation_method: the segmentation method to use. One of 'grabcut', 'cnn' or 'none'. If 'cnn' the model will use U-Net.
    dataset: the dataset to use (used to get the padding and input channel number)
    pretrained_unet: the checkpoint path to a pre-trained U-Net or None. The U-Net will be used to pre-train loc_net
      as well as the segmentation model if `segmentation_method` is 'cnn'.
    device: the device to use
  """
  is_3d = isinstance(dataset, data.scan_dataset.ScanDataset)

  if is_3d:
    loc_net = _get_unet_loc_net_3d(dataset, device, pretrained_unet)
  else:
    loc_net = _get_unet_loc_net(dataset, device, pretrained_unet)
  
  if segmentation_method == 'grabcut':
    segmentation_model = GrabCutSegmenter(padding=dataset.padding, **segmenter_kwargs)
  elif segmentation_method == 'cnn':
    if is_3d:
      unet = get_unet_3d(dataset, device, pretrained_unet)
    else:
      unet = get_unet(dataset, device, pretrained_unet)
    segmentation_model = CNNSegmenter(padding=dataset.padding, segmentation_model=unet, **segmenter_kwargs)
  elif segmentation_method == 'none':
    segmentation_model = None
  else:
    raise ValueError(f'Invalid segmentation method: {segmentation_method}')
  
  model = PreCut(loc_net=loc_net, segmentation_model=segmentation_model, pretraining=pretraining, spatial_dims=3 if is_3d else 2)
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
    loc_net: the localization network.
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
    pretraining: if `True`, the model will be used for pre-training (disables transformation and segmentation).
    stn_zoom_out: the zoom out factor for the STN. The STN will zoom out by this factor before segmentation (only when not pretraining).
    spatial_dims: the number of spatial dimensions (2 or 3)

  Returns: 
    A dictionary with keys:
      - 'img_th': thresholded image
      - 'threshold': the threshold with shape (batch_size, 2)
      - 'img_stn: STN-transformed image (without thresholding)
      - 'theta': the affine matrix output by the STN with shape (batch_size, 3, 2)
      - 'img_th_stn': STN-transformed image (with thresholding)
      - 'seg': segmentation mask obtained with GrabCut
  """
  def __init__(self, loc_net, pretraining=False, segmentation_model=None, spatial_dims=2, stn_zoom_out=1.15):
      super(PreCut, self).__init__()

      self.segmentation_model = segmentation_model
      self.pretraining = pretraining
      self.stn_zoom_out = stn_zoom_out
      self.spatial_dims = spatial_dims
      print('Pretraining:', pretraining)

      # Spatial transformer localization-network
      self.loc_net = loc_net

      if spatial_dims == 2:
        self.encoder_output_size = loc_net._out_channels[loc_net._depth] * 4 * 4
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
      elif spatial_dims == 3:
        self.encoder_output_size = 320 * 5 * 5 * 5
      else:
        raise ValueError(f'Invalid spatial_dims: {spatial_dims}, must be 2 or 3')

      # Regressor for the threshold
      self.thresh_head = nn.Sequential(
        nn.Linear(self.encoder_output_size, 128),
        nn.ReLU(True),
        nn.Linear(128, 3 * 2) # channels x (min, max)
      )

      # Initialize to untresholded image
      self.thresh_head[-1].weight.data.zero_()
      self.thresh_head[-1].bias.data.copy_(torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float))

      # Regressor for the affine matrix
      self.stn_head = nn.Sequential(
        nn.Linear(self.encoder_output_size, 128),
        nn.ReLU(True),
        nn.Linear(128, 3 * 2 if spatial_dims == 2 else 3 * 4)
      )

      # Initialize to identity transformation
      self.stn_head[-1].weight.data.zero_()
      if spatial_dims == 2:
        self.stn_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
      elif spatial_dims == 3:
        self.stn_head[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

  def transform(self, x, theta, size):
    grid = F.affine_grid(theta, size, align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)
    return x

  def smooth_threshold(self, x, low, high):
    slope = 50
    low = low[:, :, None, None]
    high = high[:, :, None, None]
    th_low = 1 / (1 + torch.exp(slope * (low - x)))
    th_high = 1 / (1 + torch.exp(-slope * (high - x)))
    return th_low + th_high - 1

  def forward(self, x):
    original_size = x.shape
    if self.spatial_dims == 2:
      xs = self.loc_net(x)[-1]
      xs = self.avg_pool(xs)
      xs = xs.view(-1, self.encoder_output_size)
    elif self.spatial_dims == 3:
      xs = self.loc_net(x)
      xs = features = xs[1:][::-1][0]
      xs = xs.view(-1, self.encoder_output_size)
    
    threshold = self.thresh_head(xs)
    threshold = threshold.view(-1, 3, 2)
    th_low = threshold[:, :, 0]
    th_high = threshold[:, :, 1]

    # Threshold the image
    if not self.pretraining:
      mask = self.smooth_threshold(x, th_low, th_high)
      x_th = x * mask
    else:
      x_th = None
    
    # Spatial transformer

    theta = self.stn_head(xs)
    if self.spatial_dims == 2:
      theta = theta.view(-1, 2, 3)
    elif self.spatial_dims == 3:
      theta = theta.view(-1, 3, 4)

    if not self.pretraining:
      # zoom out to add padding
      theta[:, 0, 0] *= self.stn_zoom_out
      theta[:, 1, 1] *= self.stn_zoom_out

      size = list(x.shape)
      #for i in range(self.spatial_dims):
      #  size[-i - 1] = size[-i - 1] // 2

      grid = F.affine_grid(theta, size, align_corners=True)

      x = F.grid_sample(x, grid, align_corners=True)
      mask = F.grid_sample(mask, grid, align_corners=True)
      x_th_stn = F.grid_sample(x_th, grid, align_corners=True)
    else:
      x_th_stn = None
    
    # print(theta)
    # plt.imshow(x_th_stn[0, 0].detach().cpu().numpy())
    # plt.show()


    row = torch.tensor([0, 0, 1] if self.spatial_dims == 2 else [0, 0, 0, 1], dtype=theta.dtype, device=theta.device)
    row = row.expand(theta.shape[0], 1, self.spatial_dims + 1)
    theta_sq = torch.cat([theta, row], dim=1)
    try:
      theta_inv = torch.inverse(theta_sq)[:, :self.spatial_dims, :]
    except:
      theta_inv = torch.zeros_like(theta_sq)[:, :self.spatial_dims, :]

    if self.segmentation_model is not None:
      # TODO: Just using original_size will result in wrong number of channels if channels > 1 for either the
      # input or mask.
      # TODO - Move back to x_th_stn
      seg = self.segmentation_model({'img_th_stn': x, 'theta_inv': theta_inv}, original_size=original_size)
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