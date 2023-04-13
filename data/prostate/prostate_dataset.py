import data.slice_dataset as slice_dataset

def ProstateDataset(subset, pretraining, **kwargs):
  return slice_dataset.SliceDataset(
    subset=subset,
    pretraining=pretraining,
    dataset_folder='prostate',
    window_max=450,
    window_min=150,
    global_min=-1024,
    global_max=1024,
    size=128,
    padding=4,
    **kwargs
  )