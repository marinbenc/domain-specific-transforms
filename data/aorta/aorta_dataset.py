import data.ct_dataset as ct_dataset
import numpy as np

class AortaDataset(ct_dataset.CTDataset):
  """
  subset: one of 'D', 'K' or 'R'
  """
  dataset_folder = 'aorta'

  WINDOW_MAX = 600
  WINDOW_MIN = 150
  GLOBAL_PIXEL_MEAN = 0.1

  GLOBAL_MIN = -200
  GLOBAL_MAX = 1000

  in_channels = 1
  out_channels = 1

  width = 256
  height = 256

  padding = 4
  th_aug = 0.05

  def __init__(self, directory, subset='D', augment=True, transforms=[]):
    super().__init__(directory, subset, augment, transforms)

    if self.subset == 'D':
      self.WINDOW_MAX = 600
      self.WINDOW_MIN = 100
    elif self.subset == 'K':
      self.WINDOW_MAX = 1100
      self.WINDOW_MIN = 800
    elif self.subset == 'R':
      self.WINDOW_MAX = 1200
      self.WINDOW_MIN = 900

