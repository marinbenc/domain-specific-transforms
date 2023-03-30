import argparse
import json
import datetime
import os
import os.path as p
from pathlib import Path
import train

def process_config(config, folder_name, dataset_percent=1.):
  config['dataset_percent'] = dataset_percent
  config['overwrite'] = True
  config['log_name'] = folder_name
  config['folds'] = 4
  return config

def main(config_folder):
  config_folder = Path(config_folder)
  folder_name = config_folder.name
  unet_config = json.load(open(config_folder / 'unet_config.json'))
  precut_config = json.load(open(config_folder / 'precut_config.json'))
  precut_unet_config = json.load(open(config_folder / 'precut_unet_config.json'))

  dataset_percentages = [0.05, 0.1, 0.25, 0.5, 1]

  print('-----------------')
  print('Training U-Net')
  print('-----------------')

  for dataset_percent in dataset_percentages:
    print(f'Running U-Net for {dataset_percent * 100}% of the dataset')
    config = process_config(unet_config, folder_name, dataset_percent)
    train.main(config)
  
  print('-----------------')
  print('Training PreCut')
  print('-----------------')

  config = process_config(precut_config, folder_name)
  train.main(config)

  print('-----------------')
  print('Training PreCut + U-Net')
  print('-----------------')

  for dataset_percent in dataset_percentages:
    print(f'Running PreCut + U-Net for {dataset_percent * 100}% of the dataset')
    config = process_config(precut_unet_config, folder_name, dataset_percent)
    train.main(config)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="""
    Run a full experiment from config files. The expriment consists of:

    1. Training an initial U-Net model on a percentage of strong labels.
    2. Training a PreCut model using the trained U-Net as encoder on strong + weak labels.
    3. Fine-tuning the combined PreCut and U-Net model on the strong labels.

    Each step is done using 4-fold cross validation. The results are saved in runs/log-name.
    For more details on each step, check `python -m train --help`.

    The config files for the experiments can be written manually but a compatible config 
    file will also get saved when running `python -m train`. Examples can be found in
    the configs/ folder.

    The config folder needs to contain the following files:
    - unet_config.json - config file for the initial U-Net model
    - precut_config.json - config file for the PreCut model
    - precut_unet_config.json - config file for the combined PreCut and U-Net model

    Note: `dataset_percent` in the config files is ignored since this script runs experiments
    for different percentages of the dataset automatically.

    The results are saved in runs/<config folder name> using the same format as `python -m train`.
    """,
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument(
    '--config-folder', type=str, help='name of folder where models are saved',
  )
  args = parser.parse_args()
  args = vars(args)
  train(**args)