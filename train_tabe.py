import random
import os
import os.path as p
import shutil
import datetime
import argparse
import functools
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import monai
import segmentation_models_pytorch as smp

import data.datasets as data
import utils
import segmenters.tabe_unet as tabe_unet

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

def get_model(device, dataset):
  base_model = smp.Unet('resnet18', in_channels=3, classes=1, 
                        activation='sigmoid', decoder_use_batchnorm=True)
  model = tabe_unet.TabeUNet(base_model=base_model, n_aux_classes=dataset.num_classes)
  model.encoder.to(device)
  model.decoder.to(device)
  model.segmentation_head.to(device)
  model.aux_head.to(device)
  return model

def train(args_object, batch_size, epochs, lr, dataset, alpha, log_name, device, 
          folds, overwrite, workers, stratified_sampling):
  def worker_init(worker_id):
    np.random.seed(2022 + worker_id)
  os.makedirs(name=f'runs/{log_name}', exist_ok=True)

  datasets = []

  if folds == 1:
    train_dataset, valid_dataset = data.get_datasets(dataset)
    datasets.append((train_dataset, valid_dataset))
    json_dict = {
      'train_subjects': list(train_dataset.subjects),
      'valid_subjects': list(valid_dataset.subjects)
    }
    with open(f'runs/{log_name}/subjects.json', 'w') as f:
      json.dump(json_dict, f)
  else:
    whole_dataset = data.get_whole_dataset(dataset)
    subject_ids = list(whole_dataset.subjects)
    subject_ids = sorted(subject_ids)

    existing_split = p.join('runs', log_name, 'subjects.json')
    if p.exists(existing_split):
      print('Using existing subject split')
      with open(existing_split, 'r') as f:
        json_dict = json.load(f)
      splits = zip(json_dict['train_subjects'], json_dict['valid_subjects'])
    else:
      kfold = KFold(n_splits=folds, shuffle=True, random_state=2022)
      splits = list(kfold.split(subject_ids))
      # convert from indices to subject ids
      splits = [([subject_ids[idx] for idx in train_idx], [subject_ids[idx] for idx in valid_idx]) for train_idx, valid_idx in splits]

      json_dict = {
        'train_subjects': [ids for (ids, _) in splits],
        'valid_subjects': [ids for (_, ids) in splits]
      }
      with open(f'runs/{log_name}/subjects.json', 'w') as f:
        json.dump(json_dict, f)

    for fold, (train_ids, valid_ids) in enumerate(splits):
      dataset_class = data.get_dataset_class(dataset)
      train_dataset = dataset_class(subset='all', subjects=train_ids, augment=True)
      valid_dataset = dataset_class(subset='all', subjects=valid_ids, augment=False)
      datasets.append((train_dataset, valid_dataset))
      # check for data leakage
      intersection = set(train_dataset.file_names).intersection(set(valid_dataset.file_names))
      assert len(intersection) == 0, f'Found {len(intersection)} overlapping files in fold {fold}'

  for fold, (train_dataset, valid_dataset) in enumerate(datasets):
    print('----------------------------------------')
    print(f'Fold {fold}')
    print('----------------------------------------')

    log_dir = f'runs/{log_name}/fold{fold}'
    if p.exists(log_dir):
      if overwrite:
        shutil.rmtree(log_dir)
      else:
        raise ValueError(f'Log directory already exists: {log_dir}. Use --overwrite to overwrite.')

    utils.save_args(args_object, log_dir)

    if stratified_sampling:
      print('Using stratified sampling.')
      train_sampler = data.lesion.StratifiedSampler(train_dataset)
      valid_sampler = data.lesion.StratifiedSampler(valid_dataset)
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init, num_workers=workers, sampler=train_sampler)
      valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init, num_workers=workers, sampler=valid_sampler)
    else:
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init, num_workers=workers)
      valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init, num_workers=workers)

    dice_loss = monai.losses.DiceLoss(include_background=False)
    bce_loss = nn.BCEWithLogitsLoss()

    model = get_model(device, train_dataset)
    
    trainer = tabe_unet.TabeTrainer(model=model, train_loader=train_loader, val_loader=valid_loader, seg_loss=dice_loss,
                                    aux_loss=bce_loss, device=device, lr=lr, momentum=0, alpha=alpha, checkpoint_name='best_model.pth', log_dir=log_dir)
    trainer.train(epochs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="""
    TODO
    """, # TODO
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
    '--folds',
    type=int,
    default=1,
    help='number of cross validation folds (1 = no cross validation)',
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=8,
    help='input batch size for training (default: 16)',
  )
  parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train (default: 100)',
  )
  parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='learning rate (default: 0.001)',
  )
  parser.add_argument(
    '--dataset', type=str, choices=data.dataset_choices, default='lesion', help='which dataset to use'
  )
  parser.add_argument(
    '--alpha',
    type=float,
    default=0.03,
    help='the weight to be applied to the confusion loss',
  )
  parser.add_argument(
    '--log-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), help='name of folder where models are saved',
  )
  parser.add_argument(
    '--overwrite', action='store_true', help='overwrite existing log folder',
  )
  parser.add_argument(
    '--device', type=str, default='cuda', help='which device to use for training',
  )
  parser.add_argument(
    '--workers',
    type=int,
    default=8,
  )
  parser.add_argument(
    '--stratified-sampling', action='store_true', help='use stratified sampling for training and validation',
  )
  # TODO: Add --from-json option to load args from json file

  args = parser.parse_args()
  args = vars(args)
  train(args_object=args, **args)