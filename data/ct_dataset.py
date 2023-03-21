import sys
import os.path as p

import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A

import data.base_dataset as base_dataset
import utils

# TODO: Rename to 3D dataset or something similar. Maybe SliceDataset?
class CTDataset(base_dataset.BaseDataset):
  """
  A dataset for CT scans.

  Attributes:
    dataset_folder: The name of the folder containing the dataset.
    WINDOW_MAX: The maximum value of the window.
    WINDOW_MIN: The minimum value of the window.

    GLOBAL_PIXEL_MEAN: The global pixel mean.

    GLOBAL_MIN: The global minimum value.
    GLOBAL_MAX: The global maximum value.

    in_channels: The number of input channels.
    out_channels: The number of output channels.

    width: The width of the image.
    height: The height of the image.

    padding: The padding to add to the image for STN transform.
    th_aug: The threshold augmentation for threshold transform.

    file_names: The list of file names.
    subjects: The list of subjects, one for each file name.
  """
  dataset_folder = None

  WINDOW_MAX = 500
  WINDOW_MIN = 200
  GLOBAL_PIXEL_MEAN = 0.1

  GLOBAL_MIN = -200
  GLOBAL_MAX = 1000

  in_channels = 1
  out_channels = 1

  width = 256
  height = 256

  padding = 4
  th_aug = 0.05

  def __init__(self, directory, augment=True, transforms=[]):
    super().__init__(directory, augment, transforms)
    self.subjects = ['_'.join(f.split('/')[-1].split('_')[:-1]) for f in self.file_names]

  def get_train_augmentation(self):
    return A.Compose([
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15),
      A.GridDistortion(p=0.5),
    ])

  def get_optimal_threshold(self, scan, mask, th_padding=0):
    if np.sum(mask) == 0:
      return scan.min(), scan.min()
    roi = scan[mask > 0]
    # TODO: Investigate this. Or use blur?
    high = np.percentile(roi, 99) + th_padding
    low = np.percentile(roi, 1) - th_padding
    return low, high

  def get_item_np(self, idx, augmentation=None):
    current_slice_file = self.file_names[idx]

    scan = np.load(current_slice_file.replace('label/', 'input/'))
    if len(scan.shape) == 3:
      # use the first channel (e.g. for prostate which is a multi-modal dataset)
      # TODO: Multi-channel support?
      scan = scan[..., 0]
    mask = np.load(current_slice_file)
    # Just use single class. TODO: Add multi-class support?
    mask[mask > 0.5] = 1

    scan[scan < self.GLOBAL_MIN] = self.GLOBAL_MIN
    scan[scan > self.GLOBAL_MAX] = self.GLOBAL_MIN

    scan = scan.astype(np.float)

    if 'th' in self.transforms:
      low, high = self.get_optimal_threshold(scan, mask, self.padding)
      self.WINDOW_MAX = high
      self.WINDOW_MIN = low

      if self.augment and self.mode == 'train' and th_aug > 0:
        self.WINDOW_MAX += np.random.randint(-high * self.th_aug, high * self.th_aug)
        self.WINDOW_MIN += np.random.randint(-high * self.th_aug, high * self.th_aug)

      # window input slice
      scan[scan > self.WINDOW_MAX] = self.WINDOW_MIN
      scan[scan < self.WINDOW_MIN] = self.WINDOW_MIN

      # normalize
      scan = (scan - self.WINDOW_MIN) / (self.WINDOW_MAX - self.WINDOW_MIN + 1e-8)
    else:
      scan = (scan - self.GLOBAL_MIN) / (self.GLOBAL_MAX - self.GLOBAL_MIN)

    if augmentation is not None:
      transformed = augmentation(image=scan, mask=mask)
      scan = transformed['image']
      mask = transformed['mask']

    if 'stn' in self.transforms:
      # TODO: Make bbox_aug a command line argument
      bbox_aug = self.padding // 2 if self.augment and self.mode == 'train' else 0
      scan, mask = utils.crop_to_label(scan, mask, bbox_aug=bbox_aug, padding=self.padding)

    return scan, mask

  def __len__(self):
    return len(self.file_names)

  def __getitem__(self, idx):
    if self.augment and self.mode == 'train':
      augmentation = self.get_train_augmentation()
    else:
      augmentation = None
    
    input, label = self.get_item_np(idx, augmentation=augmentation)

    #utils.show_images_row(imgs=[input + 0.5, label])
        
    # to PyTorch expected format
    input = np.expand_dims(input, axis=0)
    label = np.expand_dims(label, axis=0)

    input_tensor = torch.from_numpy(input).float()
    label_tensor = torch.from_numpy(label)

    #utils.show_torch([input_tensor + 0.5, label_tensor])

    return input_tensor, label_tensor



