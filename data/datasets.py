import data.lesion.lesion_dataset as lesion
import data.liver.liver_dataset as liver
import data.eat.eat_dataset as eat
import data.aorta.aorta_dataset as aorta

dataset_choices = ['lesion', 'liver', 'eat', 'aorta']
lesion_subsets = ['isic', 'dermquest', 'dermis']
aorta_subsets = ['D', 'K', 'R']
all_subsets = lesion_subsets + aorta_subsets

def get_dataset_class(dataset_name):
  if dataset_name == 'lesion':
    return lesion.LesionDataset
  elif dataset_name == 'liver':
    return liver.LiverDataset
  elif dataset_name == 'eat':
    return eat.EATDataset
  elif dataset_name == 'aorta':
    return aorta.AortaDataset
  else:
    raise ValueError(f'Unknown dataset name: {dataset_name}')

def get_datasets(dataset, subset='isic', augment=True, transforms=[]):
  if dataset == 'lesion':
    train_dataset = lesion.LesionDataset(directory='train', augment=augment, subset=subset, transforms=transforms)
    val_dataset = lesion.LesionDataset(directory='valid', subset=subset, augment=augment, transforms=transforms)
  elif dataset == 'liver':
    pass
  elif dataset == 'aorta':
    train_dataset = aorta.AortaDataset(directory='train', augment=augment, subset=subset, transforms=transforms)
    val_dataset = aorta.AortaDataset(directory='valid', subset=subset, augment=False, transforms=transforms)
  elif dataset == 'eat':
    train_dataset = eat.EATDataset(directory='train', augment=augment, transforms=transforms)
    val_dataset = eat.EATDataset(directory='valid', augment=False, transforms=transforms)
  return (train_dataset, val_dataset)

def get_whole_dataset(dataset, subset='isic', transforms=[]):
  if dataset == 'lesion':
    dataset = lesion.LesionDataset(directory='all', subset=subset, augment=False, transforms=transforms)
  elif dataset == 'liver':
    dataset = liver.LiverDataset(directory='all', augment=False, transforms=transforms)
  elif dataset == 'aorta':
    dataset = aorta.AortaDataset(directory='all', subset=subset, augment=False, transforms=transforms)
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
  elif dataset == 'aorta':
    test_dataset = aorta.AortaDataset(directory='test', subset=subset, augment=False, transforms=transforms)
  return test_dataset