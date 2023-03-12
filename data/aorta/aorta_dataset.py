import sys
import os.path as p

import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A

import data.base_dataset as base_dataset
import utils

class AortaDataset(base_dataset.BaseDataset):
  """
  subset: one of 'D', 'K' or 'R'
  """
  dataset_folder = 'aorta'

  WINDOW_MAX = 500
  WINDOW_MIN = 200
  # obtained empirically
  GLOBAL_PIXEL_MEAN = 0.1

  GLOBAL_MIN = -200
  GLOBAL_MAX = 1000

  in_channels = 1
  out_channels = 1

  width = 256
  height = 256

  padding = 4

  def get_train_transforms(self):
    return A.Compose([
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15),
      A.GridDistortion(p=0.5),
    ])

  def get_optimal_threshold(self, scan, mask, th_padding=0):
    aorta_region = scan[mask > 0]
    high = np.percentile(aorta_region, 100) + th_padding
    low = np.percentile(aorta_region, 1) - th_padding
    return low, high

  def get_item_np(self, idx, transform=None):
    current_slice_file = self.file_names[idx]

    if self.subset == 'D':
      self.WINDOW_MAX = 500
      self.WINDOW_MIN = 200
    elif self.subset == 'K':
      self.WINDOW_MAX = 1100
      self.WINDOW_MIN = 800
    elif self.subset == 'R':
      self.WINDOW_MAX = 1200
      self.WINDOW_MIN = 900

    scan = np.load(current_slice_file.replace('label/', 'input/'))
    mask = np.load(current_slice_file)

    scan[scan < self.GLOBAL_MIN] = self.GLOBAL_MIN
    scan[scan > self.GLOBAL_MAX] = self.GLOBAL_MAX

    scan = scan.astype(np.float)

    if 'itn' in self.transforms:
      aorta_region = scan[mask > 0]
      th_aug = 0.05

      low, high = self.get_optimal_threshold(scan, mask)
      self.WINDOW_MAX = high
      self.WINDOW_MIN = low

      if self.augment and self.mode == 'train' and th_aug > 0:
        self.WINDOW_MAX += np.random.randint(-high * th_aug, high * th_aug)
        self.WINDOW_MIN += np.random.randint(-high * th_aug, high * th_aug)

      # window input slice
      scan[scan > self.WINDOW_MAX] = self.WINDOW_MIN
      scan[scan < self.WINDOW_MIN] = self.WINDOW_MIN

      # plt.imshow(scan, cmap='gray')
      # plt.show()
      # normalize
      scan = (scan - self.WINDOW_MIN) / (self.WINDOW_MAX - self.WINDOW_MIN + 1e-8)
    else:
      scan = (scan - self.GLOBAL_MIN) / (self.GLOBAL_MAX - self.GLOBAL_MIN)

    if transform is not None:
      transformed = transform(image=scan, mask=mask)
      scan = transformed['image']
      mask = transformed['mask']

    if 'stn' in self.transforms:
      # TODO: Make bbox_aug a command line argument
      bbox_aug = 2 if self.augment and self.mode == 'train' else 0
      scan, mask = utils.crop_to_label(scan, mask, bbox_aug=bbox_aug, padding=self.padding)

    return scan, mask

  def __len__(self):
    return len(self.file_names)

  def __getitem__(self, idx):
    if self.augment and self.mode == 'train':
      transforms = self.get_train_transforms()
    else:
      transforms = None
    
    input, label = self.get_item_np(idx, transform=transforms)

    #utils.show_images_row(imgs=[input + 0.5, label])
        
    # to PyTorch expected format
    input = np.expand_dims(input, axis=0)
    label = np.expand_dims(label, axis=0)

    input_tensor = torch.from_numpy(input).float()
    label_tensor = torch.from_numpy(label)

    #utils.show_torch([input_tensor + 0.5, label_tensor])

    return input_tensor, label_tensor

