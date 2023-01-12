import numpy as np
import argparse
import torch

import os.path as p

from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

import pandas as pd

import utils
import seg.train_seg as seg
import data.datasets as data
import stn.stn_dataset as stn_dataset
import stn.train_stn as stn
import fine_tune

device = 'cuda'

def get_checkpoint(model_type, log_name):
  checkpoint = p.join('runs', log_name, model_type, f'{model_type}_best.pth')  
  return torch.load(checkpoint)

def get_predictions(model, dataset):
  xs = []
  ys = []
  ys_pred = []

  model.eval()
  with torch.no_grad():
    for idx, (data, target) in enumerate(dataset):
      x_np, y_np = dataset.get_item_np(idx)

      xs.append(x_np)
      ys.append(y_np)

      data, target = data.to(device), target.to(device)
      output = model(data.unsqueeze(0))
      output = F.interpolate(output, y_np.shape[-2:], mode='nearest')
      output = output.squeeze().detach().cpu().numpy()
      output = utils._thresh(output)
      ys_pred.append(output)

      #utils.show_images_row(imgs=[x_np, y_np, output], titles=['x', 'y', 'y_pred'])

  return xs, ys, ys_pred

def run_stn_predictions(model, dataset):
  model.eval()
  with torch.no_grad():
    for data, target in dataset:
      data, target = data.to(device), target.to(device)
      output = model(data.unsqueeze(0))
      utils.show_torch([data + 0.5, output.squeeze() + 0.5, (data + 0.5) - (output.squeeze() + 0.5), target.squeeze() + 0.5])


def calculate_metrics(ys_pred, ys, metrics):
  '''
  Parameters:
    ys_pred: model-predicted segmentation masks
    ys: the GT segmentation masks
    metrics: a dictionary of type `{metric_name: metric_fn}` 
    where `metric_fn` is a function that takes `(y_pred, y)` and returns a float.

  Returns:
    A DataFrame with one column per metric and one row per image.
  '''
  metric_names, metric_fns = metrics.keys(), metrics.values()
  df = pd.DataFrame(columns=metric_names)

  for (y_pred, y) in zip(ys_pred, ys):
    df.loc[len(df)] = [metric(y_pred, y) for metric in metric_fns]

  return df

def test(model_type, dataset, log_name, dataset_folder, subset):
  train_dataset, valid_dataset = data.get_datasets(dataset, subset, augment=False)
  whole_dataset = data.get_whole_dataset(dataset, subset)
  test_dataset = data.get_test_dataset(dataset, subset)

  if dataset_folder == 'train':
    test_dataset = train_dataset
  elif dataset_folder == 'valid':
    test_dataset = valid_dataset
  elif dataset_folder == 'all':
    test_dataset = whole_dataset
  
  if model_type == 'seg':
    model = seg.get_model(test_dataset)
  elif model_type == 'stn':
    model = stn.get_model(test_dataset)
    checkpoint = get_checkpoint(model_type, log_name)
    model.load_state_dict(checkpoint['model'])
    test_dataset = stn_dataset.STNDataset(test_dataset)
    run_stn_predictions(model, test_dataset)
    exit()
  if model_type == 'fine':
    model = fine_tune.get_model(test_dataset, log_name)

  checkpoint = get_checkpoint(model_type, log_name)
  model.load_state_dict(checkpoint['model'])

  #TODO: Implement testing for STN

  xs, ys, ys_pred = get_predictions(model, test_dataset)
  metrics = {
    'dsc': utils.dsc,
    'prec': utils.precision,
    'rec': utils.recall,
  }
  df = calculate_metrics(ys, ys_pred, metrics)

  print(df.describe())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test a trained model'
    )
    parser.add_argument(
        '--model-type', type=str, choices=['seg', 'stn', 'fine'], required=True, help='type of model to be tested',
    )
    parser.add_argument(
        '--dataset', type=str, choices=data.dataset_choices, default='lesion', help='which dataset to use'
    )
    parser.add_argument(
        '--subset', type=str, choices=data.lesion_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--dataset-folder', type=str, choices=['train', 'valid', 'test', 'all'], default='test'
    )
    parser.add_argument(
        '--log-name', type=str, required=True, help='name of folder where checkpoints are stored',
    )
    args = parser.parse_args()
    test(**vars(args))