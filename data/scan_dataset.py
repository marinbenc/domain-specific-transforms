import os.path as p
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
import cv2 as cv
import monai.transforms as T

import data.pre_cut_dataset as pre_cut_dataset
import utils

class ScanDataset(pre_cut_dataset.PreCutDataset):
  """
  A dataset for volumetric CT and MRI scans.

  Attributes:
    dataset_folder: The name of the folder containing the dataset.
                    The folder needs to contain train/, valid/ and test/ folders.
                    Inside, the files need to be named <subject_id>_<slice_number>.npy.
    window_max: The maximum value of the window.
    window_min: The minimum value of the window.
    global_max: The global maximum value (across all images).
    global_min: The global minimum value (across all images).
    in_channels: The number of input channels.
    out_channels: The number of output channels.
    size: The size of the images (single float). Images are assumed to be square.
    padding: The padding to add to the image for STN transform in pixels.
    subjects: A list of subject IDs to include in the dataset, or `None` to include all subjects.
  """
  # TODO: Add support for cross validation. subjects is currently ignored.
  def __init__(self, subset, pretraining, size, dataset_folder, global_max, global_min,
                window_max, window_min, in_channels=1, out_channels=1, padding=8, th_padding=0.05, 
                subjects=None, augment=False, return_transformed_img=False, manually_threshold=False, stn_zoom_out=1.125):
    super().__init__(subset, pretraining, in_channels, out_channels, size, padding, th_padding, augment, stn_zoom_out, return_transformed_img)
    self.dataset_folder = dataset_folder
    self.GLOBAL_MAX = global_max
    self.GLOBAL_MIN = global_min
    self.manual_threshold = (window_min, window_max) if manually_threshold else None

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    self.file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, directory)
      directory_files = utils.listdir(p.join(directory, 'label'))
      directory_files = [p.join(directory, 'label', f) for f in directory_files]
      directory_files.sort()
      self.file_names += directory_files
      self.file_names.sort()
    
    if subjects is not None:
      self.file_names = [f for f in self.file_names if self._get_subject_from_file_name(f) in subjects]
    
    self.subject_id_for_idx = [self._get_subject_from_file_name(f) for f in self.file_names]
    self.subjects = subjects if subjects is not None else set(self.subject_id_for_idx)
  
  def _get_subject_from_file_name(self, file_name):
    return Path(file_name).stem
  
  def get_train_augmentation(self):
    transform_kwargs = {
      'mode': ('bilinear', 'nearest'),
      'keys': ['image', 'mask'],
      'padding_mode': 'reflection',
    }

    transform = T.Compose([
      T.RandAffined(
        prob=0.5, 
        rotate_range=(0.15, 0.15, 0.15), 
        translate_range=(0.1, 0.1, 0.1), 
        scale_range=(0.1, 0.1, 0.1),
        **transform_kwargs),
      T.RandGridDistortiond(prob=0.5, distort_limit=0.15, **transform_kwargs),
    ])

    return transform
  
  def __len__(self):
    return len(self.file_names)

  def get_item_np(self, idx, augmentation=None):
    current_slice_file = self.file_names[idx]

    scan = np.load(current_slice_file.replace('label/', 'input/'))
    mask = np.load(current_slice_file)
    mask[mask > 0.5] = 1 # TODO: Add multi-class support

    # WHD -> DHW
    scan = np.transpose(scan, (2, 1, 0))
    mask = np.transpose(mask, (2, 1, 0))
    
    min = self.GLOBAL_MIN
    max = self.GLOBAL_MAX
    if self.manual_threshold is not None:
      min, max = self.manual_threshold

    scan[scan < min] = min
    scan[scan > max] = max

    scan = scan.astype(np.float)
    # normalize
    scan = (scan - min) / (max - min)

    if augmentation is not None:
      # add channel dim
      scan = np.expand_dims(scan, axis=0)
      mask = np.expand_dims(mask, axis=0)
      transformed = augmentation({'image': scan, 'mask': mask})
      scan = transformed['image'][0].numpy()
      mask = transformed['mask'][0].numpy()

    #plt.imshow(scan[64, ...])
    #plt.show()
    #plt.imshow(mask[64, ...])
    #plt.show()
    return scan, mask


