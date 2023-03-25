import data.ct_dataset as ct_dataset

class ProstateDataset(ct_dataset.CTDataset):
  dataset_folder = 'prostate'

  WINDOW_MAX = 150
  WINDOW_MIN = 450
  GLOBAL_PIXEL_MEAN = 0.1

  GLOBAL_MIN = -200
  GLOBAL_MAX = 1000

  in_channels = 1
  out_channels = 1

  width = 128
  height = 128

  padding = 0
  th_aug = 0.05
