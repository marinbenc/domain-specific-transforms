import data.ct_dataset as ct_dataset

class SpleenDataset(ct_dataset.CTDataset):
  dataset_folder = 'spleen'

  WINDOW_MAX = -100
  WINDOW_MIN = 150
  GLOBAL_PIXEL_MEAN = 0.1

  GLOBAL_MIN = -1024
  GLOBAL_MAX = 1024

  in_channels = 1
  out_channels = 1

  width = 512
  height = 512

  padding = 16
  th_aug = 0.05
