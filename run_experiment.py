import argparse
import json
import datetime
import os
import os.path as p
from pathlib import Path
import train

def process_config(config, folder_name, dataset_percent=1., train_on_transformed_imgs=False):
  config['data_percent'] = dataset_percent
  config['overwrite'] = True
  config['log_name'] = folder_name
  config['folds'] = 4
  config['train_on_transformed_imgs'] = train_on_transformed_imgs
  return config

def main(config_folder):
  config_folder = Path(config_folder)
  folder_name = config_folder.name
  unet_config = json.load(open(config_folder / 'unet_config.json'))
  precut_config = json.load(open(config_folder / 'precut_config.json'))
  precut_unet_config = json.load(open(config_folder / 'precut_unet_config.json'))

  # TODO: Add checks for which models are already trained

  # TODO: Rename dataset_percent to dataset_size, accept both percentages and number of samples
  dataset_percentages = [0.05, 0.1, 0.25, 0.5, 1.0] # TODO: Add 1.0

  print('-----------------')
  print('Training U-Net')
  print('-----------------')

  for dataset_percent in dataset_percentages:
    print(f'Running U-Net for {dataset_percent * 100}% of the dataset')
    if not (Path('runs') / folder_name / f'fold0/unet_dp={int(dataset_percent * 100)}_t=0').exists():
      config = process_config(unet_config, folder_name, dataset_percent)
      train.train(args_object=config, **config)
    else:
      print('Already trained, skipping...')

  print('-----------------')
  print('Pre-training U-Net encoder for PreCut')
  print('-----------------')

  for dataset_percent in dataset_percentages:
    print(f'Running U-Net (transformed) for {dataset_percent * 100}% of the dataset')
    if not (Path('runs') / folder_name / f'fold0/unet_dp={int(dataset_percent * 100)}_t=1').exists():
      config = process_config(unet_config, folder_name, dataset_percent, train_on_transformed_imgs=True)
      train.train(args_object=config, **config)
    else:
      print('Already trained, skipping...')
  
  print('-----------------')
  print('Training PreCut')
  print('-----------------')

  if not (Path('runs') / folder_name / f'fold0/precut_dp=100_t=0').exists():
    config = process_config(precut_config, folder_name)
    train.train(args_object=config, **config)
  else:
    print('Already trained, skipping...')

  print('-----------------')
  print('Training PreCut + U-Net')
  print('-----------------')

  for dataset_percent in dataset_percentages:
    print(f'Running PreCut + U-Net for {dataset_percent * 100}% of the dataset')
    if not (Path('runs') / folder_name / f'fold0/precut_unet_dp={int(dataset_percent * 100)}_t=0').exists():
      config = process_config(precut_unet_config, folder_name, dataset_percent)
      train.train(args_object=config, **config)
    else:
      print('Already trained, skipping...')

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
    '--config-folder', '-c', type=str, help='name of folder where models are saved',
  )
  args = parser.parse_args()
  args = vars(args)
  main(**args)