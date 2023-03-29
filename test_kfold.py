import numpy as np
import argparse
import torch
import cv2 as cv
from scipy import ndimage
import json

import matplotlib.pyplot as plt

import os
import os.path as p
from pathlib import Path

from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

import pandas as pd

import utils
import data.datasets as data
import data.pre_cut_dataset as pre_cut_dataset
import pre_cut

from test import get_checkpoint, get_predictions, calculate_metrics

device = 'cuda'

def test(model_type, dataset, log_name, dataset_folder, save_predictions, viz):
  fold_dirs = [f for f in os.listdir(p.join('runs', log_name)) if f[:4] == 'fold']
  fold_dirs.sort()
  log_name = Path(log_name)

  # load subjects from json
  with open('runs' / log_name / 'subjects.json', 'r') as f:
    subjects = json.load(f)
  
  splits = zip(subjects['train_subjects'], subjects['valid_subjects'])
  datasets = []

  for fold, fold_dir, (train_subjects, valid_subjects) in zip(range(len(fold_dirs)), fold_dirs, splits):
    dataset_class = data.get_dataset_class(dataset)
    train_dataset = dataset_class(subset='all', subjects=train_subjects, pretraining=model_type == 'precut', augment=True)
    valid_dataset = dataset_class(subset='all', subjects=valid_subjects, pretraining=False, augment=False)
    datasets.append(valid_dataset)

  xs_all = []
  ys_all = []
  ys_pred_all = []
  subjects_all = []

  for fold, test_dataset in enumerate(datasets):
    if model_type == 'unet':
      model = pre_cut.get_unet(test_dataset, device)
    elif model_type == 'precut':
      model = pre_cut.get_model(segmentation_method='grabcut', dataset=test_dataset)
    elif model_type == 'precut_unet':
      model = pre_cut.get_model(segmentation_method='cnn', dataset=test_dataset)

    checkpoint = get_checkpoint(model_type, log_name / f'fold{fold}', fold=fold)
    model.load_state_dict(checkpoint['model'])

    xs, ys, ys_pred = get_predictions(model, test_dataset, viz=viz)
    xs_all += xs
    ys_all += ys
    ys_pred_all += ys_pred
    subjects_all += test_dataset.subject_id_for_idx

  if save_predictions:
    os.makedirs(p.join('predictions', log_name), exist_ok=True)
    for i in range(len(ys_pred)):
      cv.imwrite(p.join('predictions', log_name, f'{i}.png'), ys_pred[i] * 255)
    
  metrics = {
    'dsc': utils.dsc,
    'prec': utils.precision,
    'rec': utils.recall,
  }
  df = calculate_metrics(ys, ys_pred, metrics, subjects=subjects_all)
  df = df.groupby('subject').mean()

  print(df.describe())

  return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test a trained model'
    )
    parser.add_argument(
        '--model-type', type=str, choices=['precut', 'precut_unet', 'unet'], required=True, help='type of model to be tested',
    )
    parser.add_argument(
        '--dataset', type=str, choices=data.dataset_choices, default='lesion', help='which dataset to use'
    )
    parser.add_argument(
        '--dataset-folder', type=str, choices=['train', 'valid', 'test', 'all'], default='test'
    )
    parser.add_argument(
        '--log-name', type=str, required=True, help='name of folder where checkpoints are stored',
    )
    parser.add_argument(
        '--save-predictions', action='store_true', help="save predicted images in the predictions/ folder"
    )
    parser.add_argument(
        '--viz', action='store_true', help="plot results"
    )
    args = parser.parse_args()
    test(**vars(args))