import data.slice_dataset as slice_dataset

def HeartDataset(subset, pretraining, **kwargs):
  return slice_dataset.SliceDataset(
    subset=subset,
    pretraining=pretraining,
    dataset_folder='heart',
    window_max=1200,
    window_min=700,
    global_min=0,
    global_max=2048,
    size=128,
    padding=8,
    **kwargs
  )