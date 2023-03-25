import data.lesion.lesion_dataset as lesion
import data.liver.liver_dataset as liver
import data.eat.eat_dataset as eat
import data.aorta.aorta_dataset as aorta
import data.prostate.prostate_dataset as prostate
import data.spleen.spleen_dataset as spleen

dataset_to_class = {
  'lesion': lesion.LesionDataset,
  'liver': liver.LiverDataset,
  'eat': eat.EATDataset,
  'aorta': aorta.AortaDataset,
  'prostate': prostate.ProstateDataset,
  'spleen': spleen.SpleenDataset,
}

dataset_choices = dataset_to_class.keys()

def get_dataset_class(dataset_name):
  if dataset_name not in dataset_to_class:
    raise ValueError(f'Unknown dataset {dataset_name}')
  return dataset_to_class[dataset_name]

def get_datasets(dataset, augment=True, transforms=[]):
  dataset_class = get_dataset_class(dataset)
  train_dataset = dataset_class(directory='train', augment=augment, transforms=transforms)
  val_dataset = dataset_class(directory='valid', augment=False, transforms=transforms)
  return train_dataset, val_dataset

def get_whole_dataset(dataset, transforms=[]):
  dataset_class = get_dataset_class(dataset)
  whole_dataset = dataset_class(directory='all', augment=False, transforms=transforms)
  return whole_dataset

def get_test_dataset(dataset, transforms=[]):
  dataset_class = get_dataset_class(dataset)
  test_dataset = dataset_class(directory='test', augment=False, transforms=transforms)
  return test_dataset
