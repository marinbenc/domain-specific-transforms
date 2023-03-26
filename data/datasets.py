import data.spleen.spleen_dataset as spleen

dataset_to_class = {
  'spleen': spleen.SpleenDataset,
}

dataset_choices = dataset_to_class.keys()

def get_dataset_class(dataset_name):
  if dataset_name not in dataset_to_class:
    raise ValueError(f'Unknown dataset {dataset_name}')
  return dataset_to_class[dataset_name]

def get_datasets(dataset, pretraining=False):
  dataset_class = get_dataset_class(dataset)
  train_dataset = dataset_class(subset='train', pretraining=pretraining)
  val_dataset = dataset_class(subset='valid', pretraining=False)
  return train_dataset, val_dataset

def get_whole_dataset(dataset, pretraining=False):
  dataset_class = get_dataset_class(dataset)
  whole_dataset = dataset_class(subset='all', pretraining=pretraining)
  return whole_dataset

def get_test_dataset(dataset):
  dataset_class = get_dataset_class(dataset)
  test_dataset = dataset_class(subset='test', pretraining=False)
  return test_dataset
