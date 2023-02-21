'''
Loads all of the scans from the data/ folder and saves each slice as a separate
numpy array file into the appropriate folder (e.g. train/<patient number>-<slice number>.npy) 
in the currrent directory.
'''

import os
import os.path as p
import sys

import matplotlib.pyplot as plt
import numpy as np
import nrrd
import cv2 as cv

def read_scan(file_path):
  ''' Read scan with axial view '''
  data, _ = nrrd.read(file_path)
  scan = np.rot90(data)
  scan = scan.astype(np.int16)
  return scan

scans_directory = 'data/avt/'

all_files = os.listdir(scans_directory)
all_files.sort()
all_label_files = [f for f in all_files if 'seg' in f]
np.random.seed(2022)
np.random.shuffle(all_label_files)

subsets = ['D', 'K', 'R']
for subset in subsets:
  label_files = [f for f in all_label_files if subset in f]

  folders = ('train', 'valid', 'test')

  split0 = int(len(label_files) * 0.7)
  split1 = split0 + int(len(label_files) * 0.2)

  split_label_files = (label_files[:split0], label_files[split0:split1], label_files[split1:])
  for split in split_label_files:
    print(len(split))

  for (folder, labels) in zip(folders, split_label_files):
    os.makedirs(p.join(subset, folder, 'input'), exist_ok=True)
    os.makedirs(p.join(subset, folder, 'label'), exist_ok=True)

    for mask_file in labels:
      volume_file = mask_file.replace('seg.', '')
      if not subset in volume_file:
        continue
      volume_scan = read_scan(p.join(scans_directory, volume_file))
      mask_scan = read_scan(p.join(scans_directory, mask_file))

      for i in range(mask_scan.shape[-1]):
        mask_slice = mask_scan[..., i]
        if mask_slice.sum() <= 0:
          # skip empty slices
          continue

        volume_slice = volume_scan[..., i]
        original_mask_name = mask_file.split('.')[0]

        volume_name = f'{original_mask_name}-{i}.npy'
        mask_name = f'{original_mask_name}-{i}.npy'

        volume_save_path = p.join(subset, folder, 'input', volume_name)
        mask_save_path = p.join(subset, folder, 'label', mask_name)
        print(volume_name, volume_slice.dtype)

        size = (256, 256)
        volume_slice = cv.resize(volume_slice, dsize=size, interpolation=cv.INTER_CUBIC)
        mask_slice = cv.resize(mask_slice, dsize=size, interpolation=cv.INTER_NEAREST)

        np.save(volume_save_path, volume_slice)
        np.save(mask_save_path, mask_slice)