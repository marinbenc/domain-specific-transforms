import os.path as p

import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
import cv2 as cv

import data.pre_cut_dataset as pre_cut_dataset
import utils

class ImageDataset(pre_cut_dataset.PreCutDataset):
  """
  A dataset for RGB images.

  Attributes:
    dataset_folder: The name of the folder containing the dataset.
                    The folder needs to contain train/, valid/ and test/ folders.
                    Inside, the files need to be named <subject_id>.npy.
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
  def __init__(self, subset, pretraining, size, dataset_folder, global_max, global_min,
                window_max, window_min, in_channels=1, out_channels=1, padding=8, th_padding=0.05, 
                subjects=None, augment=False, return_transformed_img=False, manually_threshold=False):
    super().__init__(subset, pretraining, in_channels, out_channels, size, padding, th_padding, augment, return_transformed_img)
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
    
    if pretraining:
      # Remove empty images for pretraining
      total_before_removal = len(self.file_names)
      to_remove = []

      for idx in range(len(self.file_names)):
        input, label = self.get_item_np(idx)
        if np.sum(label) < 5:
          to_remove.append(idx)
      
      print(f'Removing {len(to_remove)} empty images out of {total_before_removal}.')
      new_filenames = np.array(self.file_names)
      new_filenames = np.delete(new_filenames, to_remove).tolist()
      self.file_names = new_filenames

    if subjects is not None:
      self.file_names = [f for f in self.file_names if self._get_subject_from_file_name(f) in subjects]
    
    self.subject_id_for_idx = [self._get_subject_from_file_name(f) for f in self.file_names]
    self.subjects = subjects if subjects is not None else set(self.subject_id_for_idx)
  
  def _get_subject_from_file_name(self, file_name):
    return file_name.split('/')[-1].split('.')[0]
  
  def get_train_augmentation(self):
    return A.Compose([
      A.GridDistortion(p=0.5, normalized=True, border_mode=cv.BORDER_CONSTANT, value=0),
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15, border_mode=cv.BORDER_CONSTANT, value=0, rotate_method='ellipse'),
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
    if len(mask.shape) == 3 and mask.shape[0] > 1:
      mask = mask[0, ...]
    mask[mask > 0.5] = 1
    
    min = self.GLOBAL_MIN
    max = self.GLOBAL_MAX
    if self.manual_threshold is not None:
      min, max = self.manual_threshold

    scan[scan < min] = min
    scan[scan > max] = max

    scan = scan.astype(np.float)
    # normalize
    scan = (scan - min) / (max - min)
    mask = mask.astype(np.float)
    mask = mask / 255.0

    if augmentation is not None:
      transformed = augmentation(image=scan, mask=mask)
      scan = transformed['image']
      mask = transformed['mask']

    return scan, mask


