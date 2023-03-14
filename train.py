import random
import os
import os.path as p
import shutil
import datetime
import argparse
import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import data.pre_cut_dataset as pre_cut_dataset
import data.datasets as datasets
import utils
import pre_cut
import segmenters.cnn_segmenter as cnn_seg

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

def get_model(model_type, log_dir, dataset, device):
  if model_type == 'unet':
    model = pre_cut.get_unet(dataset, device)
    return model

  unet_path = p.join(log_dir, '../unet', 'unet_best.pth')
  if p.exists(unet_path):
    print('Transfer learning with: ' + unet_path)
    pretrained_unet = unet_path
  else:
    print('No saved U-Net model exists, skipping transfer learning')
    pretrained_unet = None
  
  segmentation_method = 'none' if model_type == 'precut' else 'unet'
  model = pre_cut.get_model(segmentation_method=segmentation_method, dataset=dataset, pretrained_unet=pretrained_unet)
  return model

def train(model_type, batch_size, epochs, lr, dataset, subset, threshold_loss_weight, log_name, log_dir, device):
  def worker_init(worker_id):
    np.random.seed(2022 + worker_id)

  os.makedirs(log_dir, exist_ok=True)

  if model_type == 'precut':
    dataset_class = datasets.get_dataset_class(dataset)
    train_dataset = pre_cut_dataset.PreCutDataset(dataset_class, augment=False, subset=subset, directory='train')
    valid_dataset = pre_cut_dataset.PreCutDataset(dataset_class, augment=False, subset=subset, directory='valid')
  else:
    train_dataset, valid_dataset = datasets.get_datasets(dataset, subset, augment=True)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
  valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init)
  
  model = get_model(model_type, log_dir, train_dataset, device)

  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True, min_lr=1e-15, eps=1e-15)

  if model_type == 'precut':
    loss = functools.partial(pre_cut.pre_cut_loss, threshold_loss_weight=threshold_loss_weight)
  else:
    loss = cnn_seg.DiceLoss()
  
  trainer = utils.Trainer(model, optimizer, loss, train_loader, valid_loader, 
                          log_dir=log_dir, checkpoint_name=f'{model_type}_best.pth', scheduler=scheduler)
  trainer.train(epochs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="""
    Train a PreCut model or a U-Net.

    The model checkpoints are saved in runs/log-name/model-type. 
    If runs/log-name/unet exists, PreCut will be pretrained with the U-Net encoder for --model-type precut,
    and for --model-type precut_unet, the segmentation U-Net will also be pretrained with the stored checkpoint.

    The recommended use is to first train a plain U-Net model (on strong labels), then train a 
    PreCut model using the same log-name. From there, you can generate weak labels or fine-tune a
    combined model as needed. Without any pretraining there is a chance the STN will not converge,
    or at least not as fast.

    For training PreCut preprocessing (for generating weak labels) use --model-type precut.
    For training a combined PreCut and U-Net model (for fine-tuning on strong labels) use --model-type precut_unet.
    To train a plain U-Net model (for transfer learning) use --model-type unet.
    """,
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
    '--model-type', type=str, choices=['precut', 'precut_unet', 'unet'], default='precut', 
    help='The type of model to train: precut - PreCut preprocessing; precut_unet - a combined PreCut and U-Net model; unet - just a U-Net.'
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
    '--dataset', type=str, choices=datasets.dataset_choices, default='lesion', help='which dataset to use'
  )
  parser.add_argument(
    '--threshold_loss_weight',
    type=float,
    default=5.,
    help='the weight to be applied to the threshold loss term of the PreCut loss function',
  )
  parser.add_argument(
    '--subset', type=str, choices=datasets.all_subsets, default='isic', help='which dataset to use'
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
  # TODO: Add --from-json option to load args from json file

  args = parser.parse_args()
  log_dir = f'runs/{args.log_name}/{args.model_type}'
  if p.exists(log_dir):
    if args.overwrite:
      shutil.rmtree(log_dir)
    else:
      raise ValueError(f'Log directory already exists: {log_dir}. Use --overwrite to overwrite.')

  del args.overwrite

  utils.save_args(args, args.model_type)
  args = vars(args)
  args['log_dir'] = log_dir
  train(**args)