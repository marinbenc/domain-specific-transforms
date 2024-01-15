import sys
import os
import numpy as np
from shutil import copyfile
from pathlib import Path
from glob import glob

from torch.utils.data import random_split
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.append('../../')

import PIL.Image

def get_files(folder):
  files = os.listdir(folder)
  files.sort()
  files = [f for f in files if not '.txt' in f]
  files = [os.path.join(folder, f) for f in files]
  return files

wd = Path(os.path.dirname(os.path.realpath(__file__)))

dataset_groups = ['isic']
for group in dataset_groups:
  os.makedirs(wd/group/'small'/'input', exist_ok=True)
  os.makedirs(wd/group/'small'/'label', exist_ok=True)
  os.makedirs(wd/group/'large'/'input', exist_ok=True)
  os.makedirs(wd/group/'large'/'label', exist_ok=True)

  if group == 'isic':
    VALID_INPUT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1-2_Validation_Input'
    TRAIN_INPUT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1-2_Training_Input'
    VALID_GT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1_Validation_GroundTruth'
    TRAIN_GT_FOLDER = 'downloaded_data/isic/ISIC2018_Task1_Training_GroundTruth'

    valid_input = get_files(wd/VALID_INPUT_FOLDER)
    train_input = get_files(wd/TRAIN_INPUT_FOLDER)

    valid_gt = get_files(wd/VALID_GT_FOLDER)
    train_gt = get_files(wd/TRAIN_GT_FOLDER)

    inputs = valid_input + train_input
    gts = valid_gt + train_gt
  
  elif group in ['dermis', 'dermquest']:
    subset = group.split('_')[-1]
    subset_folder = subset
    if subset_folder == 'dermis':
      subset_folder = 'dermIS'
    
    gts = glob(f'{wd}/downloaded_data/waterloo/**/*_contour.png', recursive=True)
    gts = [gt for gt in gts if subset_folder in gt or subset in gt]

    inputs = [f.replace('_contour.png', '_orig.jpg') for f in gts]

  for img_f, gt_f in zip(inputs, gts):
    gt = cv.imread(gt_f, cv.IMREAD_GRAYSCALE)
    gt = np.array(PIL.Image.fromarray(gt).resize((256, 256), PIL.Image.NEAREST)) # OpenCV INTER_NEAREST has a bug

    input = cv.imread(img_f)
    input = cv.resize(input, (256, 256), interpolation=cv.INTER_LINEAR)

    area = cv.countNonZero(gt)
    is_small = area < 8000
    scale_folder = 'small' if is_small else 'large'

    file_name = img_f.split('/')[-1]
    folder = wd/group/scale_folder
    input_destination = os.path.join(folder, 'input', file_name)
    gt_destionation = os.path.join(folder, 'label', file_name.replace('.jpg', '.png'))

    cv.imwrite(input_destination, input)
    cv.imwrite(gt_destionation, gt)
