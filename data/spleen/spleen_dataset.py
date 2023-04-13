import data.slice_dataset as slice_dataset

def SpleenDataset(subset, pretraining, **kwargs):
  return slice_dataset.SliceDataset(
    subset=subset,
    pretraining=pretraining,
    dataset_folder='spleen',
    window_max=-100,
    window_min=150,
    global_min=-100,
    global_max=150,
    size=512,
    padding=8,
    **kwargs
  )