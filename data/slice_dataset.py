import os.path as p

import numpy as np
import matplotlib.pyplot as plt

import albumentations as A

import data.pre_cut_dataset as pre_cut_dataset
import utils

class SliceDataset(pre_cut_dataset.PreCutDataset):
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
                window_max, window_min, in_channels=1, out_channels=1, padding=8, th_padding=10, subjects=None):
    super().__init__(subset, pretraining, in_channels, out_channels, size, padding, th_padding)
    self.dataset_folder = dataset_folder
    self.GLOBAL_MAX = global_max
    self.GLOBAL_MIN = global_min
    self.WINDOW_MAX = window_max
    self.WINDOW_MIN = window_min

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
  
  def _get_subject_from_file_name(self, file_name):
    return '_'.join(file_name.split('/')[-1].split('_')[:-1])
  
  def get_train_augmentation(self):
    return A.Compose([
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15),
      A.GridDistortion(p=0.5),
    ])
  
  def __len__(self):
    return len(self.file_names)

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
    scan[scan > self.GLOBAL_MAX] = self.GLOBAL_MAX

    scan = scan.astype(np.float)
    # normalize
    scan = (scan - self.GLOBAL_MIN) / (self.GLOBAL_MAX - self.GLOBAL_MIN)

    if augmentation is not None:
      transformed = augmentation(image=scan, mask=mask)
      scan = transformed['image']
      mask = transformed['mask']

    return scan, mask


