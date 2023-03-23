import sys
import os.path as p
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import utils

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

class PreCutClassificationDataset(Dataset):
  """
  A dataset that wraps another dataset to be used by the PreCut classification branch.
  The output of this dataset is a tuple of (input, is_empty), where
  is_empty outputs logits indicating whether the input slice contains any object to
  be segmented.

  Attributes:
    wrapped_dataset_class: The dataset to wrap.
    **dataset_kwargs: The arguments to pass to the dataset.
  """
  def __init__(self, wrapped_dataset_class, **dataset_kwargs):
    self.wrapped_dataset = wrapped_dataset_class(transforms=[], **dataset_kwargs)
    self.wrapped_dataset_th_stn = wrapped_dataset_class(transforms=['th', 'stn'], **dataset_kwargs)
    self.wrapped_dataset_stn = wrapped_dataset_class(transforms=['stn'], **dataset_kwargs)

    self.in_channels = self.wrapped_dataset.in_channels
    self.out_channels = self.wrapped_dataset.out_channels
    self.width = self.wrapped_dataset.width

  def get_augmentation(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.GridDistortion(p=0.5),
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15),
      # TODO: Try brightness / contrast / gamma
      ToTensorV2()
    ], additional_targets={'image_th_stn': 'image', 'image_stn': 'image'})

  def __len__(self):
    return len(self.wrapped_dataset)

  def __getitem__(self, idx):
    input, label = self.wrapped_dataset.get_item_np(idx)
    
    augmentation = self.get_augmentation()
    transformed = augmentation(image=input, mask=label)
    input = transformed['image'].float()
    label = transformed['mask']

    is_empty = (torch.sum(label) < 5).float()

    return input, is_empty

class PreCutTransformDataset(PreCutClassificationDataset):
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
  """
  def __init__(self, wrapped_dataset_class, **dataset_kwargs):
    super().__init__(wrapped_dataset_class, **dataset_kwargs)

    total_before_removal = len(self.wrapped_dataset)
    to_remove = []

    for idx in range(len(self.wrapped_dataset)):
      input, label = self.wrapped_dataset.get_item_np(idx)
      if np.sum(label) < 5:
        to_remove.append(idx)
    
    print(f'Removing {len(to_remove)} empty slices out of {total_before_removal}.')
    new_filenames = np.array(self.wrapped_dataset.file_names)
    new_filenames = np.delete(new_filenames, to_remove).tolist()

    self.wrapped_dataset.file_names = new_filenames
    self.wrapped_dataset_th_stn.file_names = new_filenames
    self.wrapped_dataset_stn.file_names = new_filenames

  def __getitem__(self, idx):
    input, label = self.wrapped_dataset.get_item_np(idx)
    is_empty = np.sum(label) < 5
    is_empty = torch.tensor(is_empty, dtype=torch.bool)

    input_th_stn, _ = self.wrapped_dataset_th_stn.get_item_np(idx)
    input_stn, _ = self.wrapped_dataset_stn.get_item_np(idx)
    th_low, th_high = self.wrapped_dataset_th_stn.get_optimal_threshold(input, label)
    th = torch.tensor([th_low, th_high], dtype=torch.float)

    augmentation = self.get_augmentation()
    transformed = augmentation(image=input, image_th_stn=input_th_stn, image_stn=input_stn, mask=label)
    input = transformed['image'].float()
    input_th_stn = transformed['image_th_stn'].float()
    input_stn = transformed['image_stn'].float()
    label = transformed['mask']

    # utils.show_torch(imgs=[input, input_stn, input_th_stn])

    return input, (input_stn, th, is_empty)
