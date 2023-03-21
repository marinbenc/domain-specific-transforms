from pathlib import Path
import json
import os
import numpy as np

import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2 as cv

def main(dataset_folder):
  dataset_folder = Path(dataset_folder)
  dsb_folder = [f for f in dataset_folder.iterdir() if f.is_dir() and f.name.startswith('Task')][0]
  dsb_folder = Path(dsb_folder)
  dsb_json = dsb_folder / 'dataset.json'
  dsb_json = json.load(dsb_json.open())

  train = dsb_json['training']
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

  global_max = 4096
  global_min = -4096

  for split, files in zip(splits, [train, valid, test]):
    save_dir = dataset_folder / split
    save_dir.mkdir(exist_ok=True)
    input_dir = save_dir / 'input'
    input_dir.mkdir(exist_ok=True)
    label_dir = save_dir / 'label'
    label_dir.mkdir(exist_ok=True)

    for file in files:
      image_file = file['image']
      label_file = file['label']

      image_file = dsb_folder / image_file
      label_file = dsb_folder / label_file

      image = sitk.ReadImage(str(image_file))
      image = sitk.GetArrayFromImage(image)

      label = sitk.ReadImage(str(label_file))
      label = sitk.GetArrayFromImage(label)
      
      case_name = image_file.name.split('/')[-1].split('.')[0]

      slices = image.shape[-1] if len(image.shape) == 3 else image.shape[-2]
      for slice_idx in range(slices):
        if len(image.shape) == 4:
          # 4D images (W, H, D, C)
          input_slice = image[..., slice_idx, :]
          label_slice = label[..., slice_idx]
        elif len(image.shape) == 3:
          # 3D images (W, H, D)
          if 'heart' in str(dataset_folder):
            input_slice = image[:, slice_idx, :]
            # crop height to width
            label_slice = label[:, slice_idx, :]
            label_slice = label[:, slice_idx, :]
            # crop height to width
            label_slice = label_slice[:, input_slice.shape[1] // 2 - input_slice.shape[0] // 2:input_slice.shape[1] // 2 + input_slice.shape[0] // 2]
            input_slice = input_slice[:, input_slice.shape[1] // 2 - input_slice.shape[0] // 2:input_slice.shape[1] // 2 + input_slice.shape[0] // 2]
            # resize to 128
            input_slice = cv.resize(input_slice, (128, 128))
            label_slice = cv.resize(label_slice, (128, 128), interpolation=cv.INTER_NEAREST)
          else:
            input_slice = image[..., slice_idx]
            label_slice = label[..., slice_idx]

          if input_slice.shape != label_slice.shape:
            print(f'file: {image_file}')
            raise ValueError(f'Input and label shape mismatch: {input_slice.shape} vs {label_slice.shape}')
          if np.sum(label_slice) == 0:
            continue
          max_value = np.percentile(input_slice[label_slice > 0.5], 99)
          min_value = np.percentile(input_slice[label_slice > 0.5], 1)
          global_max = min(global_max, max_value)
          global_min = max(global_min, min_value)
        else:
          raise ValueError(f'Invalid image shape: {image.shape}')

        np.save(input_dir / f'{case_name}_{slice_idx}.npy', input_slice)
        np.save(label_dir / f'{case_name}_{slice_idx}.npy', label_slice)

  print(f'Global max: {global_max}')
  print(f'Global min: {global_min}')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('dataset_folder', type=str, help='The folder containing the dataset')
  args = parser.parse_args()

  main(args.dataset_folder)
