import sys
import os.path as p
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2 as cv

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import utils

# TODO: Remove temp imports
import torch.nn.functional as F
import torch.nn as nn

def get_affine_from_bbox(x, y, w, h, size):
  """
  Returns an affine transformation matrix in OpenCV-expected format that
  will crop the image to the bounding box.
  """
  scale_x = size / w
  scale_y = size / h
  M = np.array([[scale_x, 0, -x * scale_x], [0, scale_y, -y * scale_y]])
  return M

def get_theta_from_bbox(x, y, w, h, size):
  """
  Returns an affine transformation matrix in PyTorch-expected format that 
  will crop the image to the bounding box.
  """
  scale_x = size / w
  scale_y = size / h

  x_t = (x + w / 2) / size * 2 - 1
  y_t = (y + h / 2) / size * 2 - 1

  theta = np.array([[1 / scale_x, 0, x_t], [0, 1 / scale_y, y_t]], dtype=np.float32)
  return theta

def get_bbox(input, label, padding=32, bbox_aug=0):
  """ 
  Get a crop bbox enclosing the label, with padding and bbox augmentation.

  Args:
    input: input image
    label: label image
    padding: padding around label
    bbox_aug: random bbox augmentation as a percentage of the bbox size, 
              each bbox parameter (x, y, w, h) is augmented 
              by a random value in [-bbox_aug * min(w, h), bbox_aug * min(w, h)]

  Returns:
    x, y, w, h: bbox parameters
  """
  original_size = label.shape[:2]

  label_th = label.copy()
  label_th[label_th > 0.5] = 1
  label_th[label_th <= 0.5] = 0
  label_th = label_th.astype(np.uint8)
  bbox = cv.boundingRect(label_th)

  x, y, w, h = bbox

  if w == 0 or h == 0 or x < 0 or y < 0:
    print('Warning: invalid bbox, using full image')
    x, y, w, h = 0, 0, *original_size
    return x, y, w, h

  if bbox_aug > 0:
    augs = np.random.uniform(-bbox_aug * min(w, h), bbox_aug * min(w, h), size=4)
    x += augs[0]
    y += augs[1]
    w += augs[2]
    h += augs[3]

  x -= padding
  y -= padding
  w += 2 * padding
  h += 2 * padding

  return x, y, w, h

def get_optimal_threshold(image, mask, th_padding=0, th_aug=0):
  """
  Returns the optimal window for the given image and mask.
  If the mask is empty, returns (min(scan), min(scan)).

  Args:
    image: The image to threshold.
    mask: The mask to use for thresholding.
    th_padding: The padding to increase the window by on each side.

  Returns:
    (low, high): The min and max values for the window.
  """
  # TODO: Add support for RGB images
  if np.sum(mask) == 0:
    return image.min(), image.min()
  roi = image[mask > 0]
  # TODO: Investigate this. Or use blur?
  high = np.percentile(roi, 99)
  low = np.percentile(roi, 1)

  window_width = high - low
  padding_value = window_width * th_padding
  low -= padding_value
  high += padding_value

  if th_aug > 0:
    th_aug_max = window_width * th_aug
    augs = np.random.uniform(-th_aug_max, th_aug_max, size=2)
    low += augs[0]
    high += augs[1]
  
  return low, high

def WeakSupervisionDataset(pre_cut_dataset, labeled_percent=0.1):
  """
  Converts the pre-cut dataset into a weakly supervised dataset.

  Attributes:
    pre_cut_dataset: The pre-cut dataset to convert.
    labeled_percent: The percentage of images to label.

  Returns:
    (labeled_dataset: PreCutDataset, unlabeled_dataset: BaseDataset)
  """
  # TODO: This no longer works.
  all_files = pre_cut_dataset.file_names
  np.random.seed(2022)
  np.random.shuffle(all_files)

  labeled_files = all_files[:int(len(all_files) * labeled_percent)]
  unlabeled_files = all_files[int(len(all_files) * labeled_percent):]

  labeled_dataset = pre_cut_dataset.wrapped_dataset

  labeled_dataset = copy.deepcopy(pre_cut_dataset.wrapped_dataset)
  labeled_dataset.file_names = labeled_files

  unlabeled_dataset = copy.deepcopy(pre_cut_dataset)
  unlabeled_dataset.file_names = unlabeled_files

  return labeled_dataset, unlabeled_dataset

class PreCutDataset(Dataset):
  """
  A dataset that wraps another dataset to be used by the PreCut thresholding + STN branches.
  This is an abstract class that should be inherited from.

  The output of this dataset is a tuple of (input, output_dict) where output_dict has the
  following keys:

    - img_th: The thresholded image.
    - img_th_stn: The thresholded image warped by the STN.
    - seg: The segmentation mask in the original image space.
    - theta: The affine transformation matrix used to warp the image.
    - threshold: The threshold used to threshold the image.
  
  If pretraining is True, `img_th` and `img_th_stn` will be `None`.

  Required methods for subclasses:
    - `__len__(self)` should return the length of the dataset.
    - `get_item_np(self, idx, augmentation)` should return a tuple of (input, label) of numpy arrays for a given index.
      - `augmentation` is an Albumentations callable transform or `None`.
    - `get_train_augmentation(self)` should return an Albumentations callable transform for training or `None`.

  Attributes:
    subset: The subset of the dataset to use. One of 'train', 'valid', 'test'.
    pretraining: Whether to use the dataset for pretraining or not. If True, `img_th` and `img_th_stn` 
                 will be `None`. In addition, the dataset will be augmented more heavily and empty 
                 slices will be removed.
    in_channels: The number of input channels.
    out_channels: The number of output channels.
    size: The size of the input images (single float). Images are assumed to be square.
    padding: The padding to add to the STN transform.
    th_padding: The padding to add to the threshold.
    return_transformed_img: Whether to return the transformed image or untransformed input.
  """
  def __init__(self, subset, pretraining, in_channels, out_channels, width, padding, th_padding, augment, stn_zoom_out=1.5, return_transformed_img=False):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.width = width
    self.padding = padding
    self.pretraining = pretraining
    self.th_padding = th_padding
    self.th_aug = 0.05 if pretraining else 0
    self.bbox_aug = 0.1 if augment and not pretraining else 0
    self.subset = subset
    self.augment = augment
    self.stn_zoom_out = stn_zoom_out
    self.return_transformed_img = return_transformed_img
  
  def get_item_np(self, idx, augmentation):
    raise NotImplementedError
  
  def get_train_augmentation(self):
    raise NotImplementedError

  def _get_augmentation_pretraining(self):
    return A.Compose([
      #A.HorizontalFlip(p=0.3),
      A.GridDistortion(p=0.5, distort_limit=0.15, normalized=True),
      #A.ShiftScaleRotate(p=0.5, rotate_limit=8, scale_limit=0.1, shift_limit=0.1),
      # TODO: Try brightness / contrast / gamma
    ])
  
  def __getitem__(self, idx):
    input, label = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    original_size = input.shape[:2]

    assert input.shape[0] == input.shape[1], 'Input must be square.'

    output_dict = {
      'img_th_stn': None,
      'img_stn': None,
      'seg': None,
      'theta': None,
      'threshold': None,
    }

    to_tensor = ToTensorV2()

    if self.pretraining:
      augmentation = self._get_augmentation_pretraining()
      transformed = augmentation(image=input, mask=label)
      input, label = transformed['image'], transformed['mask']
        
    input_tensor, label_tensor = to_tensor(image=input, mask=label).values()
    input_tensor = input_tensor.float()
    output_dict['seg'] = label_tensor.unsqueeze(0).float()

    x, y, w, h = get_bbox(input, label, padding=self.padding, bbox_aug=self.bbox_aug)
    theta = get_theta_from_bbox(x, y, w, h, size=original_size[0])
    output_dict['theta'] = torch.from_numpy(theta)

    threshold = get_optimal_threshold(input, label, th_padding=self.th_padding, th_aug=self.th_aug)
    output_dict['threshold'] = torch.tensor(threshold).float()

    if not self.pretraining:
      # Pretraining loss does not use the images, so skip the transformations to save time.
      M = get_affine_from_bbox(x, y, w, h, original_size[0])
      scale = (self.stn_zoom_out - 1) # TODO: Don't hard-code this. This needs to be the same as in pre_cut.
      M[0, 2] += scale * original_size[1]
      M[1, 2] += scale * original_size[0]
      M *= scale
      input_cropped = cv.warpAffine(input, M, original_size, flags=cv.INTER_LINEAR)
      label_cropped = cv.warpAffine(label, M, original_size, flags=cv.INTER_NEAREST)
      transformed = to_tensor(image=input_cropped, mask=label_cropped)
      input_cropped_tensor, label_cropped_tensor = transformed['image'], transformed['mask']
      output_dict['img_stn'] = input_cropped_tensor
      if self.return_transformed_img:
        output_dict['seg'] = label_cropped_tensor.unsqueeze(0).float()

      input_th_stn = input_cropped.copy()
      low, high = threshold
      input_th_stn[input_th_stn < low] = low
      input_th_stn[input_th_stn > high] = low
      input_th_stn = (input_th_stn - low) / (high - low + 1e-8)
      output_dict['img_th_stn'] = to_tensor(image=input_th_stn)['image'].float()

      #utils.show_images_row([input, input_cropped, label_cropped, input_th_stn])
    
    # print(output_dict['theta'].numpy())
    
    # test viz
    # grid = F.affine_grid(output_dict['theta'].unsqueeze(0), input_tensor.unsqueeze(0).size(), align_corners=True)
    # x = F.grid_sample(input_tensor.unsqueeze(0), grid, align_corners=True)[0]
    # mask = F.grid_sample(output_dict['seg'].unsqueeze(0), grid, align_corners=True)[0]
    # plt.imshow(x[0].numpy())
    # plt.show()
    # plt.imshow(mask[0].numpy())
    # plt.show()



    # delete None values
    output_dict = {k: v for k, v in output_dict.items() if v is not None}
    if self.return_transformed_img:    
      return output_dict['img_th_stn'], output_dict
    else:
      return input_tensor, output_dict