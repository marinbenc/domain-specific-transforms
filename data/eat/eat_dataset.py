import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import albumentations as A
import kornia as K

import utils
import data.base_dataset as base_dataset

# obtained empirically
GLOBAL_PIXEL_MEAN = 0.075

class EATDataset(base_dataset.BaseDataset):
  """
  A dataset for EAT segmentation.

  Attributes:
  """
  dataset_folder = 'eat'
  width = 128
  height = 128

  in_channels = 1
  out_channels = 1

  def get_train_transforms(self):
    return A.Compose([
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(p=0.5, rotate_limit=35, scale_limit=0.15, shift_limit=0.15),
      # elastic transform
      A.GridDistortion(p=0.5, num_steps=5, distort_limit=0.3),
    ])

  def get_item_np(self, idx, transform=None):
    file_name = self.file_names[idx]
    label_file = file_name
    input_file = file_name.replace('label/', 'input/').replace('.png', '.npy')
    input = np.load(input_file)

    # clip to air and bone
    input[input < -1000] = 0
    input[input > 1000] = 0

    # threshold to fat range
    range = (-140, -30)
    input[input < range[0]] = range[0]
    input[input > range[1]] = range[0]
    input += -range[0]
    input /= range[1] - range[0]
    input -= GLOBAL_PIXEL_MEAN

    label = cv.imread(label_file, cv.IMREAD_GRAYSCALE)
    label = label.astype(np.float32)
    label /= 255.0
    
    if transform is not None:
      transformed = transform(image=input, mask=label)
      input = transformed['image']
      label = transformed['mask']
    
    if 'stn' in self.transforms:
      bbox_aug = 32 if self.augment and self.mode == 'train' else 0
      input, label = utils.crop_to_label(input, label, bbox_aug=bbox_aug)

    return input, label


  def __getitem__(self, idx):

    if self.augment and self.mode == 'train':
      augmentations = self.get_train_transforms()
    else:
      augmentations = None

    input, label = self.get_item_np(idx, transform=augmentations)
    original_size = label.shape

    #utils.show_images_row(imgs=[input + 0.5, label])
        
    # to PyTorch expected format
    input = np.expand_dims(input, axis=0)
    label = np.expand_dims(label, axis=0)

    input_tensor = torch.from_numpy(input).float()
    if 'itn' in self.transforms:
      input_tensor = utils.itn_transform_lesion(input_tensor)

    label_tensor = torch.from_numpy(label)

    #utils.show_torch([input_tensor + 0.5, label_tensor])

    return input_tensor, label_tensor