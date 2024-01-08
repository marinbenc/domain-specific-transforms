import data.lesion.lesion_dataset as lesion
import data.liver.liver_dataset as liver

dataset_choices = ['lesion', 'liver']
lesion_subsets = ['isic', 'dermquest', 'dermis', 'ph2']

def get_datasets(dataset, subset='isic', augment=True, stn_transformed=False):
  if dataset == 'lesion':
    train_dataset = lesion.LesionDataset(directory='train', augment=augment, subset=subset, stn_transformed=stn_transformed)
    val_dataset = lesion.LesionDataset(directory='valid', subset=subset, augment=augment, stn_transformed=stn_transformed)
  elif dataset == 'liver':
    train_dataset = liver.LiverDataset(directory='train', augment=augment, stn_transformed=stn_transformed)
    val_dataset = liver.LiverDataset(directory='valid', augment=augment, stn_transformed=stn_transformed)
  return (train_dataset, val_dataset)

def get_whole_dataset(dataset, subset='isic', stn_transformed=False):
  if dataset == 'lesion':
    dataset = lesion.LesionDataset(directory='all', subset=subset, augment=False, stn_transformed=stn_transformed)
  elif dataset == 'liver':
    dataset = liver.LiverDataset(directory='all', augment=False)
  return dataset

def get_test_dataset(dataset, subset='isic', stn_transformed=False):
  if dataset == 'lesion':
    test_dataset = lesion.LesionDataset(directory='test', subset=subset, augment=False, stn_transformed=stn_transformed)
  elif dataset == 'liver':
    test_dataset = liver.LiverDataset(directory='test', augment=False, stn_transformed=stn_transformed)
  return test_dataset