import os.path as p
import json

from sklearn.model_selection import KFold

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

def get_train_dataset(dataset, subset, subjects):
  if dataset == 'lesion':
    train_dataset = lesion.LesionDataset(directory='all', augment=True, subset=subset, subjects=subjects)
  return train_dataset

def get_valid_dataset(dataset, subset, subjects):
  if dataset == 'lesion':
    valid_dataset = lesion.LesionDataset(directory='all', augment=False, subset=subset, subjects=subjects)
  return valid_dataset

def get_kfolds_datasets(dataset, subset, k, log_name):
    whole_dataset = get_whole_dataset(dataset, subset)
    subject_ids = list(whole_dataset.file_names)
    subject_ids = [p.basename(f) for f in subject_ids]
    subject_ids = sorted(subject_ids)

    existing_split = p.join('runs', log_name, 'subjects.json')
    if p.exists(existing_split):
        print('Using existing subject split')
        with open(existing_split, 'r') as f:
            json_dict = json.load(f)
            splits = zip(json_dict['train_subjects'], json_dict['valid_subjects'])
    else:
        kfold = KFold(n_splits=k, shuffle=True, random_state=2022)
        splits = list(kfold.split(subject_ids))
        # convert from indices to subject ids
        splits = [([subject_ids[idx] for idx in train_idx], [subject_ids[idx] for idx in valid_idx]) for train_idx, valid_idx in splits]
        json_dict = {
            'train_subjects': [ids for (ids, _) in splits],
            'valid_subjects': [ids for (_, ids) in splits]
        }
        with open(f'runs/{log_name}/subjects.json', 'w') as f:
            json.dump(json_dict, f)
    
    datasets = []
    for fold, (train_ids, valid_ids) in enumerate(splits):
        train_dataset = get_train_dataset(dataset, subset, train_ids)
        valid_dataset = get_valid_dataset(dataset, subset, valid_ids)
        # check for data leakage
        intersection = set(train_dataset.file_names).intersection(set(valid_dataset.file_names))
        assert len(intersection) == 0, f'Found {len(intersection)} overlapping subjects in fold {fold}'
        datasets.append((train_dataset, valid_dataset))
    
    return datasets
