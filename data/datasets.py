import data.lesion.lesion_dataset as lesion
import data.liver.liver_dataset as liver

dataset_choices = ['lesion', 'liver']
lesion_subsets = ['isic', 'dermquest', 'dermis']

def get_datasets(dataset, subset='isic', augment=True, transforms=[]):
  if dataset == 'lesion':
    train_dataset = lesion.LesionDataset(directory='train', augment=augment, subset=subset, transforms=transforms)
    val_dataset = lesion.LesionDataset(directory='valid', subset=subset, augment=augment, transforms=transforms)
  elif dataset == 'liver':
    train_dataset = liver.LiverDataset(directory='train', augment=augment, transforms=transforms)
    val_dataset = liver.LiverDataset(directory='valid', augment=augment, transforms=transforms)
  return (train_dataset, val_dataset)

def get_whole_dataset(dataset, subset='isic', transforms=[]):
  if dataset == 'lesion':
    dataset = lesion.LesionDataset(directory='all', subset=subset, augment=False, transforms=transforms)
  elif dataset == 'liver':
    dataset = liver.LiverDataset(directory='all', augment=False, transforms=transforms)
  return dataset

def get_test_dataset(dataset, subset='isic', transforms=[]):
  if dataset == 'lesion':
    test_dataset = lesion.LesionDataset(directory='test', subset=subset, augment=False, transforms=transforms)
  elif dataset == 'liver':
    test_dataset = liver.LiverDataset(directory='test', augment=False, transforms=transforms)
  return test_dataset