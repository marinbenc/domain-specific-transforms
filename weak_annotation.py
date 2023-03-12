import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt

def grab_cut_stn_output(img, theta, theta_inv, padding=0):
  '''
  Use Grab Cut to segment the image. The image is expected to be an output from thresholding and STN.
  Args:
    img: The image to segment. Must be a grayscale image with values in [0, 1].
    theta_inv: The inverse of the transformation matrix produced by the STN.
    padding: The padding used to train the STN. Used to determine the background.
  '''
  input = img.copy()
  input = cv.cvtColor(input * 255,cv.COLOR_GRAY2RGB).astype(np.uint8)

  # padding is in original image space, so scale it to the transformed image space
  scale_x, scale_y = theta_inv.squeeze()[0, 0].item(), theta_inv.squeeze()[1, 1].item()
  bbox = [
    padding // 4 * scale_x, 
    padding // 4 * scale_y, 
    input.shape[1] - padding // 2 * scale_x, 
    input.shape[0] - padding // 2 * scale_y]

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

  # transform segmentation to original image space
  grid = F.affine_grid(theta_inv, segmentation.unsqueeze(0).size(), align_corners=False).cpu()
  segmentation = F.grid_sample(segmentation.unsqueeze(0), grid)
  segmentation = segmentation.squeeze().detach().cpu().numpy().astype(np.uint8)

  return segmentation
