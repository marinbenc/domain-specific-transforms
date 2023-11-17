import sys
import os.path as p
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.ndimage as ndimage
import skimage

import monai.transforms as T
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

def get_affine_from_bbox_3d(z, y, x, d, h, w, original_shape):
  """
  Returns an affine transformation matrix in OpenCV-expected format that
  will crop the image to the bounding box.
  """
  original_h, original_w, original_d = original_shape
  scale_x = original_w / w
  scale_y = original_h / h
  scale_z = original_d / d
  M = np.array([[scale_z, 0, 0, -z * scale_z], [0, scale_x, 0, -x * scale_x], [0, 0, scale_y, -y * scale_y]])
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

def get_bbox_from_theta(theta):
  """
  Returns the bounding box parameters from an affine transformation matrix
  in PyTorch-expected format.
  """
  theta_inv = np.linalg.inv(theta)
  w = 2 / theta_inv[0, 0]
  h = 2 / theta_inv[1, 1]
  x = -theta_inv[0, 2] / theta_inv[0, 0] * w - w / 2
  y = -theta_inv[1, 2] / theta_inv[1, 1] * h - h / 2
  return x, y, w, h

def get_theta_from_bbox_3d(z, y, x, d, h, w, original_shape):
  """
  Returns an affine transformation matrix in PyTorch-expected format that 
  will crop the image to the bounding box.
  """
  original_h, original_w, original_d = original_shape
  scale_x = original_w / w
  scale_y = original_h / h
  scale_z = original_d / d

  x_t = (x + w / 2) / original_w * 2 - 1
  y_t = (y + h / 2) / original_h * 2 - 1
  z_t = (z + d / 2) / original_d * 2 - 1

  theta = np.array([[1 / scale_z, 0, 0, z_t], [0, 1 / scale_x, 0, x_t], [0, 0, 1 / scale_y, y_t]], dtype=np.float32)
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
    if label_th.sum() > 0:
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

def get_bbox_3d(input, label, padding=0, bbox_aug=0):
  """
  Get a crop bbox enclosing the label, with padding and bbox augmentation.
  Note: Expects a (D, H, W) input.

  Returns:
    z, y, x, d, h, w: bbox parameters
  """
  label_th = label.copy().squeeze()
  label_th[label_th > 0.5] = 1
  label_th[label_th <= 0.5] = 0
  label_th = label_th.astype(np.uint8)

  # use skimage to get 3d bbox
  bbox = skimage.measure.regionprops(label_th)[0].bbox
  y, x, z, h, w, d = bbox
  w = w - x
  h = h - y
  d = d - z
  
  if bbox_aug > 0:
    augs = np.random.uniform(-bbox_aug * min(w, h), bbox_aug * min(w, h), size=6)
    x += augs[0]
    y += augs[1]
    w += augs[2]
    h += augs[3]
    z += augs[4]
    d += augs[5]

  x -= padding
  y -= padding
  w += 2 * padding
  h += 2 * padding
  z -= padding
  d += 2 * padding

  # for i in range(z - 10, z + 100, 10):
  #   slice = label[..., i].astype(np.uint8).squeeze() * 255
  #   slice = np.stack([slice, slice, slice], axis=-1)
  #   slice = cv.rectangle(slice, (x, y), (x + w, y + h), (0, 255, 0), 2)
  #   plt.imshow(slice)
  #   plt.show()

  return z, y, x, d, h, w

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
      A.HorizontalFlip(p=0.3),
      A.GridDistortion(p=0.5, distort_limit=0.2, normalized=True, border_mode=cv.BORDER_CONSTANT, value=0),
      A.ShiftScaleRotate(p=0.5, rotate_limit=8, scale_limit=0.1, shift_limit=0.1, rotate_method='ellipse', border_mode=cv.BORDER_CONSTANT, value=0),
      # TODO: Try brightness / contrast / gamma
    ])

  def _get_augmentation_pretraining_3d(self):
    transform_kwargs = {
      'mode': ('bilinear', 'nearest'),
      'keys': ['image', 'mask'],
      'padding_mode': 'reflection',
    }

    transform = T.Compose([
      T.RandFlipd(prob=0.3, spatial_axis=0, keys=['image', 'mask']),
      T.RandGridDistortiond(prob=0.5, distort_limit=0.2, **transform_kwargs),
      T.RandAffined(
        prob=0.5,
        rotate_range=(0.15, 0.15, 0.15), 
        translate_range=(0.1, 0.1, 0.1), 
        scale_range=(0.1, 0.1, 0.1),
        **transform_kwargs),
    ])

    return transform

  def __getitem__(self, idx):
    input, label = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    original_size = input.shape

    #assert input.shape[0] == input.shape[1], 'Input must be square.'

    output_dict = {
      'img_th_stn': None,
      'img_stn': None,
      'seg': None,
      'theta': None,
      'threshold': None,
    }

    def to_tensor(img):
      if len(img.shape) == 2:
        img = img[None, ...]
      return torch.from_numpy(img)

    if self.pretraining:
      if len(original_size) == 2:
        augmentation = self._get_augmentation_pretraining()
        transformed = augmentation(image=input, mask=label)
        input, label = transformed['image'], transformed['mask']
      else:
        pass
        #augmentation = self._get_augmentation_pretraining_3d()
        #transformed = augmentation({'image': input, 'mask': label})
        #input = transformed['image'].numpy()
        #label = transformed['mask'].numpy()
    
    if len(input.shape) == 2:
      input_tensor, label_tensor = to_tensor(image=input, mask=label).values()
      input_tensor = input_tensor.float()
      output_dict['seg'] = label_tensor.unsqueeze(0).float()
      
      bbox = get_bbox(input, label, padding=self.padding, bbox_aug=self.bbox_aug)
      theta = get_theta_from_bbox(*bbox, size=original_size[0])
      output_dict['theta'] = torch.from_numpy(theta)
    elif len(input.shape) == 4:
      input_tensor = torch.from_numpy(input).float()
      label_tensor = torch.from_numpy(label)
      output_dict['seg'] = label_tensor.float()

      bbox = get_bbox_3d(input, label, padding=self.padding, bbox_aug=self.bbox_aug)
      theta = get_theta_from_bbox_3d(*bbox, original_size[1:])
      output_dict['theta'] = torch.from_numpy(theta)

    #threshold = get_optimal_threshold(input, label, th_padding=self.th_padding, th_aug=self.th_aug)
    output_dict['threshold'] = None#torch.tensor(threshold).float()

    # Pretraining loss does not use the images, so skip the transformations to save time.
    if not self.pretraining:
      if len(original_size) == 2:
        M = get_affine_from_bbox(*bbox, original_size[0])
        scale = (self.stn_zoom_out - 1) # TODO: Don't hard-code this. This needs to be the same as in pre_cut.
        M[0, 2] += scale * original_size[1]
        M[1, 2] += scale * original_size[0]
        M *= scale
        input_cropped = cv.warpAffine(input, M, original_size, flags=cv.INTER_LINEAR)
        label_cropped = cv.warpAffine(label, M, original_size, flags=cv.INTER_NEAREST)
      else:
        # TODO: THis is wrong. Look at theta from bbox for inspiration on how to fix.
        # z, y, x, d, h, w = bbox
        #input_cropped = input[:, y:y+h, x:y+w, z:z+d]
        #label_cropped = label[:, y:y+h, x:y+w, z:z+d]
        #scale = (self.stn_zoom_out - 1) # TODO: Don't hard-code this. This needs to be the same as in pre_cut.
        #M[0, 3] += scale * original_size[3]
        #M[1, 3] += scale * original_size[1]
        #M[2, 3] += scale * original_size[2]
        # M *= scale
        # For 3D transforms, use ndimage. However ndimage inverts the transform, so we need to invert it back.
        M = get_affine_from_bbox_3d(*bbox, original_size[1:])
        M = np.concatenate([M, np.array([[0, 0, 0, 1]])], axis=0)
        # input is currently chwd
        # input_cdhw = np.transpose(input, (0, 3, 1, 2))
        # label_dhw = np.transpose(label, (2, 0, 1))
        input_cropped = ndimage.affine_transform(input, np.linalg.inv(M), order=1)
        label_cropped = ndimage.affine_transform(label, np.linalg.inv(M), order=0)
        # slice_sums = np.array([np.sum(label_cropped[..., z]) for z in range(label_cropped.shape[-1])])
        # max_z = np.argmax(slice_sums)
        #plt.imshow(label_cropped[0, ..., max_z])
        #plt.show()

      #grid = F.affine_grid(output_dict['theta'].unsqueeze(0), input_tensor.unsqueeze(0).size(), align_corners=True)
      #x = F.grid_sample(input_tensor.unsqueeze(0), grid, align_corners=True)[0]
      #mask = F.grid_sample(output_dict['seg'].unsqueeze(0), grid, align_corners=True).squeeze()
      #if self.return_transformed_img:
      #  output_dict['seg'] = label_cropped_tensor.unsqueeze(0).float()

      #input_th_stn = input_cropped.copy()
      #low, high = threshold
      #input_th_stn[input_th_stn < low] = low
      #input_th_stn[input_th_stn > high] = low
      #input_th_stn = (input_th_stn - low) / (high - low + 1e-8)
      #output_dict['img_th_stn'] = x
      #output_dict['img_stn'] = x

      #utils.show_images_row([input, input_cropped, label_cropped, input_th_stn])
    
    # print(output_dict['theta'].numpy())


    # delete None values
    output_dict = {k: v for k, v in output_dict.items() if v is not None}
    if self.return_transformed_img:    
      return output_dict['img_th_stn'], output_dict
    else:
      return input_tensor, output_dict