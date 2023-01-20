import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv

import data.base_dataset as base_dataset

NUM_SLICES_PER_SCAN = 841
WINDOW_MAX = 200
WINDOW_MIN = 0
# obtained empirically
GLOBAL_PIXEL_MEAN = 0.1

class LiverDataset(base_dataset.BaseDataset):
  dataset_folder = 'liver'
  in_channels = 1
  out_channels = 1

  height = 128
  width = 128
  
  def get_item_np(self, idx):
    label_file = self.file_names[idx]
    input_file = label_file.replace('label/', 'input/').replace('segmentation', 'volume')
    volume_slice = np.load(input_file)
    mask_slice = np.load(label_file)
    # remove non-liver labels
    mask_slice[mask_slice > 1] = 1

    # window input slice
    volume_slice[volume_slice > WINDOW_MAX] = WINDOW_MAX
    volume_slice[volume_slice < WINDOW_MIN] = WINDOW_MIN
    
    # normalize and zero-center
    volume_slice = (volume_slice - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    # zero-centered globally because CT machines are calibrated to have even 
    # intensities across images
    volume_slice -= GLOBAL_PIXEL_MEAN

    return volume_slice, mask_slice

  def __getitem__(self, idx):
    volume_slice, mask_slice = self.get_item_np(idx)

    # convert to Pytorch expected format
    volume_slice = np.expand_dims(volume_slice, axis=-1)
    volume_slice = volume_slice.transpose(2, 0, 1)
    mask_slice = np.expand_dims(mask_slice, axis=-1)
    mask_slice = mask_slice.transpose(2, 0, 1)

    volume_tensor = torch.from_numpy(volume_slice.astype(np.float32))
    mask_tensor = torch.from_numpy(mask_slice.astype(np.float32))

    return volume_tensor, mask_tensor




  
