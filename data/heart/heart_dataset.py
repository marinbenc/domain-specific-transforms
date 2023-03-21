import data.ct_dataset as ct_dataset

class HeartDataset(ct_dataset.CTDataset):
  dataset_folder = 'heart'

  WINDOW_MAX = 700
  WINDOW_MIN = 1200
  GLOBAL_PIXEL_MEAN = 0.1

  GLOBAL_MIN = 0
  GLOBAL_MAX = 2000

  in_channels = 1
  out_channels = 1

  width = 128
  height = 128

  padding = 16
  th_aug = 0.05
