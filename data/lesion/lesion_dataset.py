import os.path as p

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import albumentations as A
import cv2 as cv
from torch.utils.data import WeightedRandomSampler

import data.pre_cut_dataset as pre_cut_dataset
import utils

def LesionDatasetISIC(subset, pretraining, **kwargs):
  return ImageDataset(subset, pretraining, 256, 'isic', global_max=255, global_min=0, window_max=255, window_min=0, **kwargs)

def LesionDatasetDermquest(subset, pretraining, **kwargs):
  return ImageDataset(subset, pretraining, 256, 'dermquest', global_max=255, global_min=0, window_max=255, window_min=0, **kwargs)

def StratifiedSampler(dataset):
  ita_df = pd.read_csv('data/lesion/dominant_colors.csv')
  ita_df['image'] = ita_df['image'].str.replace('.jpg', '')
  ita_df.set_index('image', inplace=True)

  # sort the weights by the order of the dataset
  dataset_file_names = dataset._get_files(['train', 'valid', 'test'])
  dataset_file_names = [p.basename(f).replace('.png', '') for f in dataset_file_names]
  ita_angle = np.array([ita_df.loc[f]['ita_angle'] for f in dataset_file_names])

  bins = 6
  hist = np.histogram(ita_angle, bins=bins, density=True)
  weights = hist[0] / np.sum(hist[0])
  weights = 1 / weights
  weights = weights / np.sum(weights)

  print('Stratified sampling:')
  print('Weights:', weights)
  print('Histogram:', hist[0])

  bin_per_image = np.digitize(ita_angle, hist[1], right=True)
  sample_weights = np.zeros(len(dataset))
  for i in range(len(dataset)):
    sample_weights[i] = weights[bin_per_image[i] - 1]
  
  return WeightedRandomSampler(sample_weights, len(sample_weights) * 2, replacement=True)

class ImageDataset(pre_cut_dataset.PreCutDataset):
  """
  A dataset for RGB images.

  Attributes:
    dataset_folder: The name of the folder containing the dataset.
                    The folder needs to contain train/, valid/ and test/ folders.
                    Inside, the files need to be named <subject_id>.npy.
    window_max: The maximum value of the window.
    window_min: The minimum value of the window.
    global_max: The global maximum value (across all images).
    global_min: The global minimum value (across all images).
    in_channels: The number of input channels.
    out_channels: The number of output channels.
    size: The size of the images (single float). Images are assumed to be square.
    padding: The padding to add to the image for STN transform in pixels.
    subjects: A list of subject IDs to include in the dataset, or `None` to include all subjects.
  """
  def __init__(self, subset, pretraining, size, dataset_folder, global_max, global_min,
                window_max, window_min, in_channels=3, out_channels=1, padding=16, th_padding=0.1, 
                subjects=None, augment=False, return_transformed_img=False, manually_threshold=False, colorspace='rgb'):
    super().__init__(subset, pretraining, in_channels, out_channels, size, padding, th_padding, augment, return_transformed_img)
    self.dataset_folder = dataset_folder
    self.GLOBAL_MAX = global_max
    self.GLOBAL_MIN = global_min
    self.manual_threshold = (window_min, window_max) if manually_threshold else None
    self.colorspace = colorspace

    assert self.colorspace in ['lab', 'rgb']

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    self.file_names = self._get_files(directories)
    
    if pretraining:
      # Remove empty images for pretraining
      total_before_removal = len(self.file_names)
      to_remove = []

      for idx in range(len(self.file_names)):
        input, label = self.get_item_np(idx)
        if np.sum(label) < 5:
          to_remove.append(idx)
      
      print(f'Removing {len(to_remove)} empty images out of {total_before_removal}.')
      new_filenames = np.array(self.file_names)
      new_filenames = np.delete(new_filenames, to_remove).tolist()
      self.file_names = new_filenames

    if subjects is not None:
      self.file_names = [f for f in self.file_names if self._get_subject_from_file_name(f) in subjects]
    
    self.subject_id_for_idx = [self._get_subject_from_file_name(f) for f in self.file_names]
    self.subjects = subjects if subjects is not None else set(self.subject_id_for_idx)
  
  def _get_files(self, directories):
    file_names = []
    for directory in directories:
      directory = p.join(p.dirname(__file__), self.dataset_folder, directory)
      directory_files = utils.listdir(p.join(directory, 'label'))
      directory_files = [p.join(directory, 'label', f) for f in directory_files]
      directory_files.sort()
      file_names += directory_files
      file_names.sort()
    return file_names


  def _get_subject_from_file_name(self, file_name):
    return file_name.split('/')[-1].split('.')[0]
  
  def get_train_augmentation(self):
    return A.Compose([
      A.Flip(p=0.4),
      A.ShiftScaleRotate(p=0.4, rotate_limit=90, scale_limit=0.1, shift_limit=0.1, border_mode=cv.BORDER_CONSTANT, value=0, rotate_method='ellipse'),
      A.GridDistortion(p=0.4, border_mode=cv.BORDER_CONSTANT, value=0)
    ])
  
  def __len__(self):
    return len(self.file_names)

  def get_item_np(self, idx, augmentation=None):
    current_file = self.file_names[idx]

    input = cv.imread(current_file.replace('label/', 'input/').replace('.png', '.jpg'))
    # convert input to LAB
    if self.colorspace == 'lab':
      input = cv.cvtColor(input, cv.COLOR_BGR2LAB)
    elif self.colorspace == 'rgb':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    input = input.transpose(2, 0, 1)

    mask = cv.imread(current_file, cv.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float)
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    min = self.GLOBAL_MIN
    max = self.GLOBAL_MAX
    if self.manual_threshold is not None:
      min, max = self.manual_threshold

    input[input < min] = min
    input[input > max] = max

    input = input.astype(np.float)
    # normalize
    input = (input - min) / (max - min)

    if augmentation is not None:
      input = input.transpose(1, 2, 0)
      transformed = augmentation(image=input, mask=mask)
      input = transformed['image']
      input = input.transpose(2, 0, 1)
      mask = transformed['mask']

    return input, mask

def show_image(img, mask=None):
  if img.shape[0] == 3:
    img_ = img.transpose(1, 2, 0)
  img_ = cv.cvtColor((img_ * 255).astype(np.uint8), cv.COLOR_LAB2RGB)
  if mask is not None:
    mask_ = np.stack([mask, mask, mask], axis=2) * 255
    mask_ = mask_.astype(np.uint8)
    plt.imshow(cv.addWeighted(img_, 0.75, mask_, 0.25, 0))
  else:
    plt.imshow(img_)
  plt.show()