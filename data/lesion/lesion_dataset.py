import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import albumentations as A

import torch.nn.functional as F

import utils

class LesionDataset(Dataset):
  """
  A dataset for skin lesion segmentation.

  Attributes:
    directory: The directory to load the dataset from. One of 'train', 'test', 'valid' or 'all'.
    subset: The subset of the dataset to load. One of 'isic', 'dermis', 'dermquest'.
    augment: Whether to augment the dataset.
  """

  width = 256
  height = 256

  in_channels = 3
  out_channels = 1

  def __init__(self, directory, subset='isic', augment=True):
    self.mode = directory
    self.augment = augment
    self.subset = subset

    if directory == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [directory]

    self.file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), subset, directory)
      directory_files = utils.listdir(p.join(directory, 'label'))
      directory_files = [p.join(directory, 'label', f) for f in directory_files]
      directory_files.sort()
      self.file_names += directory_files

  def get_train_transforms(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.ShiftScaleRotate(p=0.5, rotate_limit=45, scale_limit=0.2, shift_limit=0.2)
    ])

  def get_item_np(self, idx):
    """
    Gets the raw unprocessed item in as a numpy array.
    """
    file_name = self.file_names[idx]
    label_file = file_name
    input_file = file_name.replace('label/', 'input/').replace('.png', '.jpg')

    label = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
    label = label.astype(np.float32)
    label /= 255.0
    
    input = cv.imread(input_file)
    input = cv.cvtColor(input, cv.COLOR_BGR2RGB)

    return input, label
    
  def __len__(self):
    length = len(self.file_names)
    return length

  def __getitem__(self, idx):
    input, label = self.get_item_np(idx)

    input = input.astype(np.float32)
    input /= 255.0
    input -= 0.5

    if self.augment and self.mode == 'train':
      transforms = self.get_train_transforms()
      transformed = transforms(image=input, mask=label)
      input = transformed['image']
      label = transformed['mask']
    
    # to PyTorch expected format
    input = input.transpose(2, 0, 1)
    label = np.expand_dims(label, axis=-1)
    label = label.transpose(2, 0, 1)

    input_tensor = torch.from_numpy(input)
    label_tensor = torch.from_numpy(label)

    #utils.show_torch([input_tensor + 0.5, label_tensor])

    return input_tensor, label_tensor
