import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import albumentations as A

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
  width = 512
  height = 512

  in_channels = 3
  out_channels = 1

  def get_train_transforms(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.ShiftScaleRotate(p=0.5, rotate_limit=45, scale_limit=0.15, shift_limit=0.15)
    ])

  def get_item_np(self, idx, transform=None):
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

    input = input.astype(np.float32)
    input /= 255.0
    input -= 0.5

    if transform is not None:
      transformed = transform(image=input, mask=label)
      input = transformed['image']
      label = transformed['mask']

    if self.stn_transformed:
      bbox_aug = 32 if self.augment and self.mode == 'train' else 0
      input, label = utils.crop_to_label(input, label, bbox_aug=bbox_aug)

    return input, label

  def __getitem__(self, idx):
    if self.augment and self.mode == 'train':
      transforms = self.get_train_transforms()
    else:
      transforms = None
    
    input, label = self.get_item_np(idx, transform=transforms)
    original_size = label.shape

    #utils.show_images_row(imgs=[input + 0.5, label])
        
    # to PyTorch expected format
    input = input.transpose(2, 0, 1)
    label = np.expand_dims(label, axis=-1)
    label = label.transpose(2, 0, 1)

    input_tensor = torch.from_numpy(input)
    label_tensor = torch.from_numpy(label)

    #utils.show_torch([input_tensor + 0.5, label_tensor])

    return input_tensor, label_tensor
