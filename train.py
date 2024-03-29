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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import monai

import data.pre_cut_dataset as pre_cut_dataset
import data.datasets as data
import utils
import pre_cut
import segmenters.cnn_segmenter as cnn_seg

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

def get_model(model_type, log_dir, dataset, device, fold, data_percent, is_transformed):
  if model_type == 'unet':
    model = pre_cut.get_unet(dataset, device)
    return model

  unet_path = p.join(log_dir, f'../unet_dp={int(data_percent * 100)}_t=1', f'unet_best_fold={fold}.pth')
  if model_type == 'precut':
    # precut is pretrained on untransformed images
    unet_path = p.join(log_dir, f'../unet_dp={int(data_percent * 100)}_t=0', f'unet_best_fold={fold}.pth')

  print(unet_path)
  if p.exists(unet_path):
    print('Transfer learning with: ' + unet_path)
    pretrained_unet = unet_path
  else:
    print('No saved U-Net model exists, skipping transfer learning')
    pretrained_unet = None
  
  segmentation_method = 'none' if model_type == 'precut' else 'cnn'

  # to pretrain precut_unet always use 100% version of precut (weakly labeled)
  precut_path = p.join(log_dir, f'../precut_dp=100_t=0', f'precut_best_fold={fold}.pth')
  pretrained_precut = precut_path if model_type == 'precut_unet' else None
  model = pre_cut.get_model(segmentation_method=segmentation_method, 
                            dataset=dataset, pretrained_unet=pretrained_unet, 
                            pretrained_precut=pretrained_precut, pretraining=model_type == 'precut')

  return model

def train(args_object, model_type, batch_size, epochs, lr, dataset, threshold_loss_weight, log_name, device, folds, 
          data_percent, overwrite, train_on_transformed_imgs, workers):
  def worker_init(worker_id):
    np.random.seed(2022 + worker_id)
  os.makedirs(name=f'runs/{log_name}', exist_ok=True)

  datasets = []

  if folds == 1:
    train_dataset, valid_dataset = data.get_datasets(dataset, pretraining=model_type == 'precut', return_transformed_img=train_on_transformed_imgs)
    datasets.append((train_dataset, valid_dataset))
    json_dict = {
      'train_subjects': list(train_dataset.subjects),
      'valid_subjects': list(valid_dataset.subjects)
    }
    with open(f'runs/{log_name}/subjects.json', 'w') as f:
      json.dump(json_dict, f)
  else:
    whole_dataset = data.get_whole_dataset(dataset, pretraining=model_type == 'precut')
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

    if data_percent < 1 and model_type != 'precut':
      new_splits = []
      for train_ids, valid_ids in splits:
        remaining_train_ids = train_ids[:int(len(train_ids) * data_percent)]
        print(f'Using {len(remaining_train_ids)} out of {len(train_ids)} training subjects')
        new_splits.append((remaining_train_ids, valid_ids))
      splits = new_splits

    for fold, (train_ids, valid_ids) in enumerate(splits):
      dataset_class = data.get_dataset_class(dataset)
      train_dataset = dataset_class(subset='all', subjects=train_ids, pretraining=model_type == 'precut', 
                                    augment=True, return_transformed_img=train_on_transformed_imgs, manually_threshold=model_type == 'unet')
      valid_dataset = dataset_class(subset='all', subjects=valid_ids, pretraining=False, augment=model_type == 'precut', 
                                    return_transformed_img=train_on_transformed_imgs, manually_threshold=model_type == 'unet')
      datasets.append((train_dataset, valid_dataset))
      # check for data leakage
      intersection = set(train_dataset.file_names).intersection(set(valid_dataset.file_names))
      assert len(intersection) == 0, f'Found {len(intersection)} overlapping files in fold {fold}'

  for fold, (train_dataset, valid_dataset) in enumerate(datasets):
    print('----------------------------------------')
    print(f'Fold {fold}')
    print('----------------------------------------')

    log_dir = f'runs/{log_name}/fold{fold}/{model_type}_dp={int(data_percent * 100)}_t={int(train_on_transformed_imgs)}'
    if p.exists(log_dir):
      if overwrite:
        shutil.rmtree(log_dir)
      else:
        raise ValueError(f'Log directory already exists: {log_dir}. Use --overwrite to overwrite.')

    utils.save_args(args_object, log_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init, num_workers=workers)
    valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init, num_workers=workers)

    if model_type == 'unet':
      dice_loss = monai.losses.DiceLoss(include_background=False)
      def loss_fn(pred, target):
        target = target['seg']
        return dice_loss(pred, target)
      loss = loss_fn
    elif model_type == 'precut':
      loss = functools.partial(pre_cut.pre_cut_loss, threshold_loss_weight=threshold_loss_weight)
    elif model_type == 'precut_unet':
      loss = cnn_seg.DiceLoss()

    model = get_model(model_type, log_dir, train_dataset, device, fold, data_percent, train_on_transformed_imgs)
      
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True, min_lr=1e-15, eps=1e-15)
    
    trainer = utils.Trainer(model, optimizer, loss, train_loader, valid_loader, 
                            log_dir=log_dir, checkpoint_name=f'{model_type}_best_fold={fold}.pth', scheduler=scheduler)
    trainer.train(epochs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="""
    Train a PreCut model or a U-Net.

    The model checkpoints are saved in runs/log-name/model-type. 
    If runs/log-name/unet exists, PreCut will be pretrained with the U-Net encoder for --model-type precut,
    and for --model-type precut_unet, the segmentation U-Net will also be pretrained with the stored checkpoint.

    The recommended use is to first train a plain U-Net model (on strong labels), then train a 
    PreCut model using the same log-name. From there, you can fine-tune a
    combined model. Without any pretraining there is a chance the STN will not converge,
    or at least not as fast.

    For pre-training PreCut preprocessing use --model-type precut.
    For training a combined PreCut and U-Net model (for fine-tuning on strong labels) use --model-type precut_unet.
    To train a plain U-Net model (for transfer learning) use --model-type unet.
    """,
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
    '--model-type', type=str, choices=['precut', 'precut_unet', 'unet'], default='precut', 
    help=
    """The type of model to train: 
      precut - PreCut preprocessing; 
      precut_unet - a combined PreCut and U-Net model (with frozen PreCut weights);
      unet - just a U-Net."""
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
    '--threshold-loss-weight',
    type=float,
    default=200.,
    help='the weight to be applied to the threshold loss term of the PreCut loss function when model type is precut',
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
    '--data-percent',
    type=float,
    default=1.,
    help='percentage of data to use for training (default: 1.0)',
  )
  parser.add_argument(
    '--train-on-transformed-imgs', action='store_true', help="train on transformed images"
  )
  parser.add_argument(
    '--workers', type=int, default=8, help='number of workers for data loading',
  )
  # TODO: Add --from-json option to load args from json file

  args = parser.parse_args()
  args = vars(args)
  train(args_object=args, **args)