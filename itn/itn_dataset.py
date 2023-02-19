import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import kornia as K

import torch.nn.functional as F

import data.lesion.lesion_dataset as lesion_dataset

import utils

def ITNDatasetLesion(**kwargs):
  return ITNDataset(lesion_dataset.LesionDataset(**kwargs), utils.itn_transform_lesion)

class ITNDataset(Dataset):
  """
  A dataset that wraps another dataset to be used by the ITN network.

  Attributes:
    wrapped_dataset: The dataset to wrap.
  """
  def __init__(self, wrapped_dataset, transforms, augment=False, segmentation=False):
    # TODO: Refactor segmentation to lesiondataset
    self.transforms = transforms
    self.wrapped_dataset = wrapped_dataset
    self.augment = augment
    self.segmentation = segmentation

  def get_augmentation(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.RandomBrightnessContrast(p=0.5),
      A.ColorJitter(p=0.5),
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

    if self.augment:
      augmentation = self.get_augmentation()
      transformed = augmentation(image=input_np, mask=label_np)
      input_np = transformed['image']
      label_np = transformed['mask']

    orig_transformed = ToTensorV2()(image=input_np, mask=label_np)
    input = orig_transformed['image']
    label = orig_transformed['mask']
    transformed = self.transforms(input)


    #utils.show_torch(imgs=[input + 0.5, input_cropped + 0.5, label])

    if self.segmentation:
      return transformed, label
    else:
      return input, transformed



    