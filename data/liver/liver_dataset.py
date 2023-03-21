import data.ct_dataset as ct_dataset

class LiverDataset(ct_dataset.CTDataset):
  dataset_folder = 'liver'

  WINDOW_MAX = 550
  WINDOW_MIN = 150
  GLOBAL_PIXEL_MEAN = 0.1

  in_channels = 1
  out_channels = 1

  width = 256
  height = 256

  padding = 4
  th_aug = 0.05

