import data.spleen.spleen_dataset as spleen
import data.prostate.prostate_dataset as prostate
import data.heart.heart_dataset as heart
import data.kidney.kidney_dataset as kidney
import data.lesion.lesion_dataset as lesion

dataset_to_class = {
  'spleen': spleen.SpleenDataset,
  'prostate': prostate.ProstateDataset,
  'heart': heart.HeartDataset,
  'kidney': kidney.KidneyDataset,
  'kidney_male': kidney.KidneyDatasetMale,
  'lesion_isic': lesion.LesionDatasetISIC,
  'lesion_dermquest': lesion.LesionDatasetDermQuest,
  'lesion_dermis': lesion.LesionDatasetDermis,
}

dataset_choices = dataset_to_class.keys()

def get_dataset_class(dataset_name):
  if dataset_name not in dataset_to_class:
    raise ValueError(f'Unknown dataset {dataset_name}')
  return dataset_to_class[dataset_name]

def get_datasets(dataset,):
  dataset_class = get_dataset_class(dataset)
  train_dataset = dataset_class(subset='train', augment=True)
  val_dataset = dataset_class(subset='valid', augment=False)
  return train_dataset, val_dataset

def get_whole_dataset(dataset, pretraining=False):
  dataset_class = get_dataset_class(dataset)
  whole_dataset = dataset_class(subset='all')
  return whole_dataset

def get_test_dataset(dataset):
  dataset_class = get_dataset_class(dataset)
  test_dataset = dataset_class(subset='test', augment=False)
  return test_dataset
