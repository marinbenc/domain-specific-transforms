import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import utils
import data.base_dataset as base_dataset

class LesionDataset(base_dataset.BaseDataset):
  """
  A dataset for skin lesion segmentation.

  Attributes:
    directory: The directory to load the dataset from. One of 'train', 'test', 'valid' or 'all'.
    subset: The subset of the dataset to load. One of 'isic', 'dermis', 'dermquest'.
    augment: Whether to augment the dataset.
  """
  dataset_folder = 'lesion'
  width = 256
  height = 256

  in_channels = 3
  out_channels = 1

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

  def __getitem__(self, idx):
    input, label = self.get_item_np(idx)
    original_size = label.shape

    input = input.astype(np.float32)
    input /= 255.0
    input -= 0.5

    if self.stn_transformed:
      input, label = utils.crop_to_label(input, label)

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
