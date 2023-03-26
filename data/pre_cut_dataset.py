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
    bbox_aug: random bbox augmentation in pixels, 
              each bbox parameter (x, y, w, h) is augmented 
              by a random value in [-bbox_aug, bbox_aug]

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

  if bbox_aug > 0:
    augs = np.random.randint(-bbox_aug, bbox_aug, size=4)
    x += augs[0]
    y += augs[1]
    w += augs[2]
    h += augs[3]

  x -= padding
  y -= padding
  w += 2 * padding
  h += 2 * padding

  return x, y, w, h

def get_optimal_threshold(self, image, mask, th_padding=0, th_aug=0):
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
  if np.sum(image) == 0:
    return image.min(), image.min()
  roi = image[mask > 0]
  # TODO: Investigate this. Or use blur?
  high = np.percentile(roi, 99) + th_padding
  low = np.percentile(roi, 1) - th_padding
  if th_aug > 0:
    th_aug_max = (high - low) * th_aug
    augs = np.random.randint(-th_aug_max, th_aug_max, size=2)
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
  # TODO: Make CTDataset and others inherit from this instead of wrapping a dataset.
  """
  A dataset that wraps another dataset to be used by the PreCut thresholding + STN branches.
  The output of this dataset is a tuple of (input, (input_stn, th)), where
  input_stn is the input image transformed by the STN, and th is the
  optimal threshold for the input image.

  Note:
    Empty slices are removed from the dataset.

  Attributes:
    wrapped_dataset_class: The dataset to wrap.
    **dataset_kwargs: The arguments to pass to the dataset.
    pretraining: Whether to use the pretraining dataset. The pretraining dataset
      removes empty slices and uses heavy augmentation to pre-train the STN and
      thresholding branches.
  """
  def __init__(self, wrapped_dataset_class, pretraining, **dataset_kwargs):
    self.wrapped_dataset = wrapped_dataset_class(transforms=[], augment=pretraining, **dataset_kwargs)
    self.in_channels = self.wrapped_dataset.in_channels
    self.out_channels = self.wrapped_dataset.out_channels
    self.width = self.wrapped_dataset.width
    self.padding = self.wrapped_dataset.padding
    self.subjects = self.wrapped_dataset.subjects
    self.pretraining = pretraining
    self.th_padding = self.wrapped_dataset.th_padding
    self.th_aug = 0.05 if pretraining else 0
    self.bbox_aug = self.padding // 2 if pretraining else 0

    if pretraining:
      # Remove empty slices for pretraining
      total_before_removal = len(self.wrapped_dataset)
      to_remove = []

      for idx in range(len(self.wrapped_dataset)):
        input, label, _ = self.wrapped_dataset.get_item_np(idx)
        if np.sum(label) < 5:
          to_remove.append(idx)
      
      print(f'Removing {len(to_remove)} empty slices out of {total_before_removal}.')
      new_filenames = np.array(self.wrapped_dataset.file_names)
      new_filenames = np.delete(new_filenames, to_remove).tolist()

      self.wrapped_dataset.file_names = new_filenames
      self.wrapped_dataset_th_stn.file_names = new_filenames
      self.wrapped_dataset_stn.file_names = new_filenames

  def get_augmentation_pretraining(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.GridDistortion(p=0.5),
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15, border_mode=cv.BORDER_CONSTANT, value=0),
      # TODO: Try brightness / contrast / gamma
    ])
  
  def __len__(self):
    return len(self.wrapped_dataset)

  def __getitem__(self, idx):
    input, label = self.wrapped_dataset.get_item_np(idx)
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
      augmentation = self.get_augmentation_pretraining()
      transformed = augmentation(image=input, mask=label)
      input, label = transformed['image'], transformed['mask']
    
    output_dict['seg'] = to_tensor(label)
    input_tensor = to_tensor(input)

    x, y, w, h = get_bbox(input, label, padding=self.padding, bbox_aug=self.bbox_aug)
    theta = get_theta_from_bbox(x, y, w, h, size=original_size[0])
    output_dict['theta'] = torch.from_numpy(theta)

    threshold = get_optimal_threshold(input, label, th_padding=self.th_padding, aug=self.th_aug)
    output_dict['threshold'] = torch.from_numpy(threshold)

    if not self.pretraining:
      # Pretraining loss does not use the images, so skip the transformations to save time.
      M = get_affine_from_bbox(x, y, w, h, original_size[0])
      input_cropped = cv.warpAffine(input, M, original_size, flags=cv.INTER_LINEAR)
      label_cropped = cv.warpAffine(label, M, original_size, flags=cv.INTER_NEAREST)
      output_dict['img_stn'] = input_cropped

      input_th_stn = input_cropped.copy()
      low, high = threshold
      input_th_stn[input_th_stn < low] = 0
      input_th_stn[input_th_stn > high] = 0
      output_dict['img_th_stn'] = to_tensor(input_th_stn)

      # utils.show_torch(imgs=[input, input_stn, input_th_stn])
    
    return input_tensor, output_dict
