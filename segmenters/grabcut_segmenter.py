import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class GrabCutSegmenter(nn.Module):
  """
  GrabCut Segmentation Model

  Used as the segmentation module of the model for weak annotation.

  Note: Since the GrabCut algorithm is not differentiable, this model is not differentiable either.

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

    for batch in range(img.shape[0]):
      input = img[batch].detach().cpu().numpy().squeeze()
      has_object = (input.max() - input.min()) > 0.1

      # padding is in original image space, so scale it to the transformed image space
      scale_x, scale_y = theta_inv.squeeze()[0, 0].item(), theta_inv.squeeze()[1, 1].item()
      bbox = [
        self.padding // 4 * scale_x, 
        self.padding // 4 * scale_y, 
        input.shape[1] - self.padding // 2 * scale_x, 
        input.shape[0] - self.padding // 2 * scale_y]

      if has_object and self._is_valid_bbox(bbox, input.shape):
        input = cv.cvtColor(input * 255,cv.COLOR_GRAY2RGB).astype(np.uint8)

        #print(bbox)
        #plt.imshow(input)
        #plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2))
        #plt.show()

        mask = np.zeros(input.shape[:2],np.uint8)
        # TODO: Check influence of circle
        # draw circle in center of bbox
        #cv.circle(mask, center=(int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)), radius=int(bbox[2] / 8), color=cv.GC_FGD, thickness=-1)
        #plt.imshow(mask)
        #plt.show()

        bg_model = np.zeros((1,65),np.float64)
        fg_model = np.zeros((1,65),np.float64)

        cv.grabCut(input, mask, bbox, bg_model, fg_model, 5, cv.GC_INIT_WITH_RECT)
          
        segmentation = np.where((mask==2)|(mask==0),0,1)
        segmentation = torch.from_numpy(segmentation).unsqueeze(0).float()
      else:
        mask = np.zeros(input.shape[:2],np.uint8)
        segmentation = torch.from_numpy(mask).unsqueeze(0).float()
    
      img[batch][:] = segmentation[:]
  
      # transform segmentation to original image space
      if has_object:
        grid = F.affine_grid(theta_inv, img.size(), align_corners=False)
        img = F.grid_sample(img, grid)
    return img

