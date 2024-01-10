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
import fine_tune
import isic_challenge_scoring as isic
from isic_challenge_scoring.load_image import ImagePair

device = 'cuda'

def get_checkpoint(model_type, log_name, fold):
  checkpoint = p.join('runs', log_name, model_type, f'fold_{fold}', f'{model_type}_best.pth')  
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
      output = F.interpolate(output, y_np.shape[-2:], mode='nearest')
      output = output.squeeze().detach().cpu().numpy()
      output = utils._thresh(output)
      ys_pred.append(output)

      #utils.show_images_row(imgs=[x_np + 0.5, data.detach().cpu().numpy().transpose(1, 2, 0) + 0.5, y_np, target.squeeze().detach().cpu().numpy(), output])

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

def isic_metric(ys_pred, y):
  ys_pred = ys_pred.astype(np.uint8)
  y = y.astype(np.uint8)
  ys_pred = ys_pred * 255
  y = y * 255

  cm = isic.confusion.create_binary_confusion_matrix(
    truth_binary_values=y > 128,
    prediction_binary_values=ys_pred > 128,
    name='0')
  
  th_jacc = isic.metrics.binary_threshold_jaccard(cm, threshold=0.65)
  return th_jacc

def test(model_type, log_name, fold, test_dataset, save_predictions):
  if model_type == 'stn':
    model = stn.get_model(test_dataset)
    checkpoint = get_checkpoint(model_type, log_name, fold)
    model.load_state_dict(checkpoint['model'])
    test_dataset = stn_dataset.STNDataset(test_dataset)
    run_stn_predictions(model, test_dataset)
    exit()

  if model_type == 'stn-to-seg':
    model = fine_tune.get_model(test_dataset, log_name, fold)
    stn_checkpoint = get_checkpoint('stn', log_name, fold)
    seg_checkpoint = get_checkpoint('seg', log_name, fold)
    model.stn.load_state_dict(stn_checkpoint['model'])
    model.seg.load_state_dict(seg_checkpoint['model'])
  else:
    if model_type == 'seg':
      model = seg.get_model(test_dataset)
    elif model_type == 'fine':
      model = fine_tune.get_model(test_dataset, log_name, fold)
      model.output_stn_mask = False

    checkpoint = get_checkpoint(model_type, log_name, fold)
    model.load_state_dict(checkpoint['model'])

  model.to(device)
  xs, ys, ys_pred = get_predictions(model, test_dataset)

  if save_predictions:
    os.makedirs(p.join('predictions', log_name, subset), exist_ok=True)
    for i in range(len(ys_pred)):
      cv.imwrite(p.join('predictions', log_name, subset, f'{i}.png'), ys_pred[i] * 255)
    
  metrics = {
    'dsc': utils.dsc,
    'prec': utils.precision,
    'rec': utils.recall,
    'th_jacc': isic_metric
  }
  df = calculate_metrics(ys, ys_pred, metrics)

  return df

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, required=True, choices=['seg', 'fine'])
  parser.add_argument('--log_name', type=str, required=True)
  parser.add_argument('--dataset', type=str, required=True, choices=data.dataset_choices)
  parser.add_argument('--subset', type=str, required=True, choices=data.lesion_subsets)
  parser.add_argument('--out_of_sample', action='store_true')
  parser.add_argument('--save_suffix', type=str, default='')
  args = parser.parse_args()

  if args.out_of_sample:
    test_dataset = data.get_valid_dataset(args.dataset, args.subset, subjects='all')
    df = pd.DataFrame()
    for fold in range(5):
      df_ = test(
        model_type=args.model_type,
        log_name=args.log_name,
        fold=fold,
        test_dataset=test_dataset,
        save_predictions=False)
      if fold == 0:
        df = df_
      else:
        df = df + df_

    # Get average across folds
    df = df / 5
  else:
    datasets = data.get_kfolds_datasets(args.dataset, args.subset, 5, args.log_name)
    df = pd.DataFrame()
    for fold, (_, test_dataset) in enumerate(datasets):
      df_ = test(
        model_type=args.model_type,
        log_name=args.log_name,
        fold=fold,
        test_dataset=test_dataset,
        save_predictions=False)
      df = pd.concat([df, df_])
  
  print(df.describe())
  df.to_csv(f'results/{args.model_type}_train={args.log_name}_test={args.subset}_{args.save_suffix}.csv')