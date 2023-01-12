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
      ToTensorV2()
    ])

  def __len__(self):
    return len(self.wrapped_dataset)

  def __getitem__(self, idx):
    input, label = self.wrapped_dataset[idx]
    input_np = input.numpy().transpose(1, 2, 0)
    label_np = label.squeeze().numpy()

    label_transforms = self.get_label_transforms()
    transformed = label_transforms(image=input_np, mask=label_np)
    input = transformed['image']
    label = transformed['mask']

    label_th = label.squeeze().detach().numpy()
    label_th[label_th > 0.5] = 1
    label_th[label_th <= 0.5] = 0
    label_th = label_th.astype(np.uint8)
    bbox = cv.boundingRect(label_th)

    padding = 32
    x, y, w, h = bbox
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w + 2 * padding, label_th.shape[1] - x)
    h = min(h + 2 * padding, label_th.shape[0] - y)

    input_cropped = input[:, y:y+h, x:x+w].clone()
    input_cropped = F.interpolate(input_cropped.unsqueeze(0), input.shape[1:], mode=cv.INTER_LINEAR)[0]

    # Augment input transformations
    input_transforms = self.get_input_transforms()
    input = input_transforms(image=input.numpy().transpose(1, 2, 0))['image']

    #utils.show_torch(imgs=[input + 0.5, input_cropped + 0.5, label])

    return input, input_cropped



    