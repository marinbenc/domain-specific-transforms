import os.path as p
import shutil

from seg.train_seg import train_seg
from fine_tune import fine_tune

default_params = {
  'epochs': 100,
  'dataset': 'lesion',
  'folds': 5,
}

params_seg = [
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 8,
    'subset': 'ph2',
  },
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 16,
    'subset': 'dermquest',
  },
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 16,
    'subset': 'dermis',
  },
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 16,
    'subset': 'isic',
  },
]

params_fine = [
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 8,
    'subset': 'ph2',
  },
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 16,
    'subset': 'dermquest',
  },
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 16,
    'subset': 'dermis',
  },
  {
    **default_params,
    'lr': 0.0001,
    'batch_size': 16,
    'subset': 'isic',
  },
]

for (param_seg, param_fine) in zip(params_seg, params_fine):
  log_name = f'{param_seg["subset"]}_final'
  log_dir = p.join('runs', log_name)
  if p.exists(log_dir):
    shutil.rmtree(log_dir)
  param_seg['log_name'] = log_name
  param_fine['log_name'] = log_name
  train_seg(**param_seg)
  fine_tune(**param_fine)
