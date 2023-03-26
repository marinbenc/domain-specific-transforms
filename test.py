import numpy as np
import argparse
import torch
import cv2 as cv
from scipy import ndimage

import matplotlib.pyplot as plt

import os
import os.path as p

from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

import pandas as pd

import utils
import data.datasets as data
import data.pre_cut_dataset as pre_cut_dataset
import pre_cut

device = 'cuda'

def get_checkpoint(model_type, log_name):
  checkpoint = p.join('runs', log_name, model_type, f'{model_type}_best.pth')
  checkpoint = torch.load(checkpoint, map_location=device)
  return checkpoint

def get_predictions(model, dataset, viz=True):
  xs = []
  ys = []
  ys_pred = []

  model.eval()
  with torch.no_grad():
    for idx, (data, target) in enumerate(dataset):
      y = target['seg']
      x_np = data.squeeze().detach().cpu().numpy()
      y_np = y.squeeze().detach().cpu().numpy()
      # get largest connected component using ndimage
      labels = ndimage.label(y_np)[0]
      y_np = (labels == np.argmax(np.bincount(labels.flat)[1:])+1).astype(int)

      xs.append(x_np)
      ys.append(y_np)

      data = data.to(device)
      output = model(data.unsqueeze(0))

      if isinstance(model, pre_cut.PreCut):
        segmentation = output['seg'].squeeze().detach().cpu().numpy()
        # post process
        segmentation = utils._thresh(segmentation)
        # if segmentation.sum() > 5:
        #   segmentation = cv.morphologyEx(segmentation, cv.MORPH_CLOSE, np.ones((3, 3)))
        #   segmentation = ndimage.binary_fill_holes(segmentation).astype(int)
        #   # get largest connected component using ndimage
        #   labels = ndimage.label(segmentation)[0]
        #   segmentation = (labels == np.argmax(np.bincount(labels.flat)[1:])+1).astype(int)

        ys_pred.append(segmentation)

        if viz and y_np.sum() > 5:
          viz_titles = ['target']
          viz_images = [target['seg'].squeeze()]

          for key, value in output.items():
            if value.dim() == 4:
              viz_titles.append(key)
              viz_images.append(value.squeeze())
          
          utils.show_torch(imgs=viz_images, titles=viz_titles)
      else:
        output = output.squeeze().detach().cpu().numpy()
        output = utils._thresh(output)
        ys_pred.append(output)

  return xs, ys, ys_pred

def calculate_metrics(ys_pred, ys, metrics, subjects=None):
  '''
  Parameters:
    ys_pred: model-predicted segmentation masks
    ys: the GT segmentation masks
    metrics: a dictionary of type `{metric_name: metric_fn}` 
    where `metric_fn` is a function that takes `(y_pred, y)` and returns a float.
    subjects: a list of subject IDs, one for each element in ys_pred. If provided, the
    returned DataFrame will have a column with the subject IDs.

  Returns:
    A DataFrame with one column per metric and one row per image.
  '''
  metric_names, metric_fns = list(metrics.keys()), metrics.values()
  columns = metric_names + ['subject']
  df = pd.DataFrame(columns=columns)

  if subjects is None:
    subjects = ['none'] * len(ys_pred)

  for (y_pred, y, subject) in zip(ys_pred, ys, subjects):
    df.loc[len(df)] = [metric(y_pred, y) for metric in metric_fns] + [subject]

  return df

def test(model_type, dataset, log_name, dataset_folder, save_predictions, viz):
  train_dataset, valid_dataset = data.get_datasets(dataset)
  whole_dataset = data.get_whole_dataset(dataset)
  test_dataset = data.get_test_dataset(dataset)

  if dataset_folder == 'train':
    test_dataset = train_dataset
  elif dataset_folder == 'valid':
    test_dataset = valid_dataset
  elif dataset_folder == 'all':
    test_dataset = whole_dataset

  if model_type == 'unet':
    model = pre_cut.get_unet(test_dataset, device)
  elif model_type == 'precut':
    model = pre_cut.get_model(segmentation_method='grabcut', dataset=test_dataset)
  elif model_type == 'precut_unet':
    model = pre_cut.get_model(segmentation_method='cnn', dataset=test_dataset)

  checkpoint = get_checkpoint(model_type, log_name)
  model.load_state_dict(checkpoint['model'])

  xs, ys, ys_pred = get_predictions(model, test_dataset, viz=viz)

  if save_predictions:
    os.makedirs(p.join('predictions', log_name), exist_ok=True)
    for i in range(len(ys_pred)):
      cv.imwrite(p.join('predictions', log_name, f'{i}.png'), ys_pred[i] * 255)
    
  metrics = {
    'dsc': utils.dsc,
    'prec': utils.precision,
    'rec': utils.recall,
  }
  df = calculate_metrics(ys, ys_pred, metrics, subjects=test_dataset.subject_id_for_idx)
  if test_dataset.subject_id_for_idx is not None:
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