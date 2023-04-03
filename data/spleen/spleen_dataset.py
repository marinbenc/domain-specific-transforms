import data.slice_dataset as slice_dataset

def SpleenDataset(subset, pretraining, subjects=None, augment=False, return_transformed_img=False):
  return slice_dataset.SliceDataset(
    subset=subset,
    pretraining=pretraining,
    dataset_folder='spleen',
    window_max=-100,
    window_min=150,
    global_min=-1024,
    global_max=1024,
    size=512,
    padding=8,
    subjects=subjects,
    augment=augment,
    return_transformed_img=return_transformed_img,
  )