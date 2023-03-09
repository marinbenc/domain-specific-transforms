import numpy as np
import argparse
import torch
import cv2 as cv

import os
import os.path as p

from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

import pandas as pd

import utils
import seg.train_seg as seg
import data.datasets as data
import stn.stn_dataset as stn_dataset
import stn.train_stn as stn
import itn.train_itn as itn
import fine_tune

device = 'cuda'

def get_checkpoint(model_type, log_name):
  checkpoint = p.join('runs', log_name, model_type, f'{model_type}_best.pth')  
  return torch.load(checkpoint)

# def get_stn_to_seg_predictions(stn_model, seg_model, dataset):

#   xs = []
#   ys = []
#   ys_pred = []

#   stn_model.eval()
#   seg_model.eval()
#   with torch.no_grad():
#     for idx, (data, target) in enumerate(dataset):
#       x_np, y_np = dataset.get_item_np(idx)
      
#       data, target = data.to(device), target.to(device)
#       stn_output = stn_model(data.unsqueeze(0))
#       seg_output = seg_model(stn_output)
#       seg_output = F.interpolate(seg_output, data.shape[-2:], mode='nearest')

#       utils.show_torch([data + 0.5, stn_output.squeeze() + 0.5, seg_output[0], target.squeeze() + 0.5])

#       seg_output = seg_output.squeeze().detach().cpu().numpy()
#       seg_output = utils._thresh(seg_output)

#       # TODO: Reverse transform

#       # TODO: Scrap this, use model, load stn checkpoint for stn and seg checkpoint for seg, don't load fine tune checkpoint.
#       !!
#       utils.show_images_row([seg_output, y_np])

#       xs.append(x_np)
#       ys.append(y_np)
#       ys_pred.append(seg_output)

#   return xs, ys, ys_pred

      

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
      if type(output) == dict:
        output = output['seg']
      output = F.interpolate(output, y_np.shape[-2:], mode='nearest')
      output = output.squeeze().detach().cpu().numpy()

      kernel = np.ones((3,3),np.uint8)
      output = cv.morphologyEx(output, cv.MORPH_OPEN, kernel)
      output = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)

      output = utils._thresh(output)
      ys_pred.append(output)

      viz_data = data.detach().cpu().numpy().transpose(1, 2, 0)
      viz_target = target.squeeze().detach().cpu().numpy()

      #utils.show_images_row(
      #  imgs=[x_np + 0.5, viz_data, viz_target, output, viz_target - output],figsize=(20, 5))

  return xs, ys, ys_pred

def run_stn_predictions(model, dataset):
  model.eval()
  with torch.no_grad():
    for data, target in dataset:
      data, target = data.to(device), target.to(device)
      output = model(data.unsqueeze(0))
      #utils.show_torch([data + 0.5, output.squeeze() + 0.5, (data + 0.5) - (output.squeeze() + 0.5), target.squeeze() + 0.5])


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

def test(model_type, dataset, log_name, dataset_folder, subset, transforms, save_predictions):
  train_dataset, valid_dataset = data.get_datasets(dataset, subset, augment=False, transforms=transforms)
  whole_dataset = data.get_whole_dataset(dataset, subset, transforms=transforms)
  test_dataset = data.get_test_dataset(dataset, subset, transforms=transforms)

  if dataset_folder == 'train':
    test_dataset = train_dataset
  elif dataset_folder == 'valid':
    test_dataset = valid_dataset
  elif dataset_folder == 'all':
    test_dataset = whole_dataset

  if model_type == 'stn':
    model = stn.get_model(test_dataset)
    checkpoint = get_checkpoint(model_type, log_name)
    model.load_state_dict(checkpoint['model'])
    test_dataset = stn_dataset.STNDataset(test_dataset)
    run_stn_predictions(model, test_dataset)
    exit()

  if model_type == 'stn-to-seg':
    model = fine_tune.get_model(test_dataset, log_name)
    stn_checkpoint = get_checkpoint('stn', log_name)
    seg_checkpoint = get_checkpoint('seg', log_name)
    model.itn = None
    model.stn.load_state_dict(stn_checkpoint['model'])
    model.seg.load_state_dict(seg_checkpoint['model'])
  if model_type == 'itn-to-seg':
    model = fine_tune.get_model(test_dataset, log_name)
    itn_checkpoint = get_checkpoint('itn', log_name)
    seg_checkpoint = get_checkpoint('seg', log_name)
    model.itn.load_state_dict(itn_checkpoint['model'])
    model.stn = None
    model.seg.load_state_dict(seg_checkpoint['model'])
  else:
    if model_type == 'seg':
      model = seg.get_model(test_dataset)
    elif model_type == 'fine':
      model = fine_tune.get_model(test_dataset, log_name)
    elif model_type == 'itn':
      model = itn.get_model(test_dataset)
      model.output_img = True
      model.output_theta = False
      model.segmentation_mode = True

    checkpoint = get_checkpoint(model_type, log_name)
    model.load_state_dict(checkpoint['model'])

  xs, ys, ys_pred = get_predictions(model, test_dataset)

  if save_predictions:
    os.makedirs(p.join('predictions', log_name, subset), exist_ok=True)
    for i in range(len(ys_pred)):
      cv.imwrite(p.join('predictions', log_name, subset, f'{i}.png'), ys_pred[i] * 255)
    
  metrics = {
    'dsc': utils.dsc,
    'prec': utils.precision,
    'rec': utils.recall,
  }
  df = calculate_metrics(ys, ys_pred, metrics)

  print(df.describe())

  return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test a trained model'
    )
    parser.add_argument(
        '--model-type', type=str, choices=['seg', 'stn', 'itn', 'fine', 'stn-to-seg', 'itn-to-seg'], required=True, help='type of model to be tested',
    )
    parser.add_argument(
        '--dataset', type=str, choices=data.dataset_choices, default='lesion', help='which dataset to use'
    )
    parser.add_argument(
        '--subset', type=str, choices=data.all_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--dataset-folder', type=str, choices=['train', 'valid', 'test', 'all'], default='test'
    )
    parser.add_argument(
        '--log-name', type=str, required=True, help='name of folder where checkpoints are stored',
    )
    parser.add_argument('--transforms', default=[], nargs='*', help='list of transformations for preprocessing; possible values: stn, itn')
    parser.add_argument(
        '--save-predictions', action='store_true', help="save predicted images in the predictions/ folder"
    )
    args = parser.parse_args()
    test(**vars(args))