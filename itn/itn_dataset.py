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
  def __init__(self, wrapped_dataset_class, subset='', directory='train'):
    self.wrapped_dataset = wrapped_dataset_class(directory, subset=subset, augment=False, transforms=[])
    self.wrapped_dataset_itn = wrapped_dataset_class(directory, subset=subset, augment=False, transforms=['itn'])

  def get_augmentation(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.GridDistortion(p=0.5),
      A.ShiftScaleRotate(p=0.5, rotate_limit=15, scale_limit=0.15, shift_limit=0.15),
      A.RandomBrightnessContrast(p=1, brightness_by_max=False),
      A.RandomGamma(p=1),
      ToTensorV2()
    ], additional_targets={'image_itn': 'image'})

  def __len__(self):
    return len(self.wrapped_dataset)

  def __getitem__(self, idx):
    input, label = self.wrapped_dataset.get_item_np(idx)
    input_itn, _ = self.wrapped_dataset_itn.get_item_np(idx)

    augmentation = self.get_augmentation()
    transformed = augmentation(image=input, image_itn=input_itn, mask=label)
    input = transformed['image'].float()
    input_itn = transformed['image_itn'].float()
    label = transformed['mask']

    #utils.show_torch(imgs=[input, input_itn, label])

    return input, input_itn
    