import data.lesion.lesion_dataset as lesion
import data.liver.liver_dataset as liver
import data.eat.eat_dataset as eat

dataset_choices = ['lesion', 'liver', 'eat']
lesion_subsets = ['isic', 'dermquest', 'dermis']

def get_datasets(dataset, subset='isic', augment=True, transforms=[]):
  if dataset == 'lesion':
    train_dataset = lesion.LesionDataset(directory='train', augment=augment, subset=subset, transforms=transforms)
    val_dataset = lesion.LesionDataset(directory='valid', subset=subset, augment=augment, transforms=transforms)
  elif dataset == 'liver':
    pass
  elif dataset == 'eat':
    train_dataset = eat.EATDataset(directory='train', augment=augment, transforms=transforms)
    # Add augmentation to the validation set because it is too small
    val_dataset = eat.EATDataset(directory='valid', augment=False, transforms=transforms)
  return (train_dataset, val_dataset)

def get_whole_dataset(dataset, subset='isic', transforms=[]):
  if dataset == 'lesion':
    dataset = lesion.LesionDataset(directory='all', subset=subset, augment=False, transforms=transforms)
  elif dataset == 'liver':
    dataset = liver.LiverDataset(directory='all', augment=False, transforms=transforms)
  elif dataset == 'eat':
    dataset = eat.EATDataset(directory='all', augment=False, transforms=transforms)
  return dataset

def get_test_dataset(dataset, subset='isic', transforms=[]):
  if dataset == 'lesion':
    test_dataset = lesion.LesionDataset(directory='test', subset=subset, augment=False, transforms=transforms)
  elif dataset == 'liver':
    test_dataset = liver.LiverDataset(directory='test', augment=False, transforms=transforms)
  elif dataset == 'eat':
    test_dataset = eat.EATDataset(directory='test', augment=False, transforms=transforms)
  return test_dataset