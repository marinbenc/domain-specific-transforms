import sys
import os.path as p

import torch
from torch.utils.data import Dataset
import albumentations as A

import utils

class BaseDataset(Dataset):
  """
  An abstract base class for datasets.

  Attributes:
    directory: The directory to load the dataset from. One of 'train', 'test', 'valid' or 'all'.
    augment: Whether to augment the dataset.
    transforms: possible values: 'stn' - GT STN transform as preprocessing, 'itn' - GT ITN transform as preprocessing.
  """

  dataset_folder = None
  width = 256
  height = 256

  in_channels = 1
  out_channels = 1

  def __init__(self, directory, subset='', augment=True, transforms=[]):
    self.mode = directory
    self.augment = augment
    self.subset = subset
    self.transforms = transforms

    if directory == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [directory]

    self.file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, subset, directory)
      directory_files = utils.listdir(p.join(directory, 'label'))
      directory_files = [p.join(directory, 'label', f) for f in directory_files]
      directory_files.sort()
      self.file_names += directory_files
      self.file_names.sort()

  def get_item_np(self, idx):
    """
    Gets the raw unprocessed item in as a numpy array.
    """
    raise NotImplementedError('get_item_np() not implemented')
    
  def __len__(self):
    length = len(self.file_names)
    return length

  def __getitem__(self, idx):
    """
    Gets the raw unprocessed item in as a numpy array.
    """
    raise NotImplementedError('get_item_np() not implemented')
