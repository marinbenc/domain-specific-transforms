import data.lesion.lesion_dataset as lesion

dataset_choices = ['lesion']
lesion_subsets = ['isic', 'dermquest', 'dermis']

def get_datasets(dataset, subset='isic', augment=True):
  if dataset == 'lesion':
    train_dataset = lesion.LesionDataset(directory='train', augment=augment, subset=subset)
    val_dataset = lesion.LesionDataset(directory='valid', subset=subset, augment=augment)
  return (train_dataset, val_dataset)

def get_whole_dataset(dataset, subset='isic'):
  if dataset == 'lesion':
    dataset = lesion.LesionDataset(directory='all', subset=subset, augment=False)
  return dataset

def get_test_dataset(dataset, subset='isic'):
  if dataset == 'lesion':
    test_dataset = lesion.LesionDataset(directory='test', subset=subset, augment=False)
  return test_dataset