import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import utils

class PreCutDataset(Dataset):
  """
  A dataset that wraps another dataset to be used by the PreCut network.
  The output of this dataset is a tuple of (input, (input_stn, th)), where
  input_stn is the input image transformed by the STN, and th is the
  optimal threshold for the input image.

  Attributes:
    wrapped_dataset_class: The dataset to wrap.
    subset: The subset of the dataset to use.
    directory: The directory to use.
  """
  def __init__(self, wrapped_dataset_class, **dataset_kwargs):
    self.wrapped_dataset = wrapped_dataset_class(transforms=[], **dataset_kwargs)
    self.wrapped_dataset_th_stn = wrapped_dataset_class(transforms=['th', 'stn'], **dataset_kwargs)
    self.wrapped_dataset_stn = wrapped_dataset_class(transforms=['stn'], **dataset_kwargs)
    self.in_channels = self.wrapped_dataset.in_channels
    self.out_channels = self.wrapped_dataset.out_channels

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

    #utils.show_torch(imgs=[input, input_itn, label])

    return input, (input_stn, th)
    