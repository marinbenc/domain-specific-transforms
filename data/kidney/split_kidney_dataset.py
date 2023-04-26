from pathlib import Path
import json
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom
import cv2 as cv

from tqdm import tqdm

def process_file(data_folder, file, input_dir, label_dir):
  image_file = file
  case_name = str.join('_', image_file.split('_')[:2])

  label_file = case_name + '.nii.gz'

  image_file = data_folder / 'imagesTr' / image_file
  label_file = data_folder / 'labelsTr' / label_file

  image = sitk.ReadImage(str(image_file))
  image = sitk.GetArrayFromImage(image)

  label = sitk.ReadImage(str(label_file))
  label = sitk.GetArrayFromImage(label)

  target_width = 186
  target_height = 186
  target_depth = 186

  image = zoom(image, (target_depth / image.shape[0], target_height / image.shape[1], target_width / image.shape[2]), order=2)
  label = zoom(label, (target_depth / label.shape[0], target_height / label.shape[1], target_width / label.shape[2]), order=0, mode='nearest')

  max_value = np.percentile(image[label > 0.5], 99)
  min_value = np.percentile(image[label > 0.5], 1)

  print(f'Case: {case_name}')

  np.save(input_dir / f'{case_name}.npy', image)
  np.save(label_dir / f'{case_name}.npy', label)

  return image.min(), image.max(), min_value, max_value



def main():
  dataset_folder = Path('')
  data_folder = Path('Dataset135_KiTS')
  train = os.listdir(data_folder / 'imagesTr')
  train.sort()

  np.random.seed(2022)
  np.random.shuffle(train)

  split0 = int(len(train) * 0.7)
  split1 = split0 + int(len(train) * 0.15)
  train, valid, test = train[:split0], train[split0:split1], train[split1:]

  print('Split info')
  print(f'Train: {len(train)}')
  print(f'Valid: {len(valid)}')
  print(f'Test: {len(test)}')

  splits = ['train', 'valid', 'test']

  mask_max = 4096
  mask_min = -4096

  global_max = 4096
  global_min = -4096

  for split, files in zip(splits, [train, valid, test]):
    save_dir = dataset_folder / split
    save_dir.mkdir(exist_ok=True)
    input_dir = save_dir / 'input'
    input_dir.mkdir(exist_ok=True)
    label_dir = save_dir / 'label'
    label_dir.mkdir(exist_ok=True)

    print(len(files))

    with ProcessPoolExecutor(max_workers=8) as executor:
      futures = [executor.submit(process_file, data_folder, file, input_dir, label_dir) for file in files]
    for future in concurrent.futures.as_completed(futures):
      min_value, max_value, mask_min_scan, mask_max_scan = future.result()
      mask_max = min(mask_max, mask_min_scan)
      mask_min = max(mask_min, mask_max_scan)
      global_max = min(global_max, max_value)
      global_min = max(global_min, min_value)

  print(f'Mask max: {mask_max}')
  print(f'Mask min: {mask_min}')
  print(f'Global max: {global_max}')
  print(f'global min: {global_min}')


if __name__ == '__main__':
  main()