import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch.nn.functional as F

import utils

class STNDataset(Dataset):
  """
  A dataset that wraps another dataset to be used by the STN network.

  Attributes:
    wrapped_dataset: The dataset to wrap.
  """
  def __init__(self, wrapped_dataset):
    self.wrapped_dataset = wrapped_dataset

  def get_input_transforms(self):
    return A.Compose([
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.01, shift_limit=0.1),
      ToTensorV2()
    ])

  def get_label_transforms(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
    ])

  def __len__(self):
    return len(self.wrapped_dataset)

  def __getitem__(self, idx):
    input, label = self.wrapped_dataset[idx]
    if input.shape[0] == 1:
      input_np = input.squeeze().numpy()
    else:
      input_np = input.numpy().transpose(1, 2, 0)
    label_np = label.squeeze().numpy()

    # TODO: Refactor this to be similar to the aorta dataset,
    # i.e. transforms inside lesion dataset and STN like ITN dataset.

    label_transforms = self.get_label_transforms()
    transformed = label_transforms(image=input_np, mask=label_np)
    input_np = transformed['image']
    label_np = transformed['mask']
    label = ToTensorV2()(image=input_np, mask=label_np)['mask']
    label = label.unsqueeze(0)

    input_cropped, _ = utils.crop_to_label(input_np, label_np)
    input_cropped = ToTensorV2()(image=input_cropped)['image']

    
    # Augment input transformations
    input_transforms = self.get_input_transforms()
    input = input_transforms(image=input_np)['image']

    #utils.show_torch(imgs=[input + 0.5, input_cropped + 0.5, label])

    return input, (input_cropped, label)



    