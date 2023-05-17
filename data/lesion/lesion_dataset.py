import os.path as p
import warnings
# color.lab2rgb() warns when values are clipped, but that is not a problem here
warnings.filterwarnings('ignore', message='.*values that have been clipped.*', append=True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from skimage import color
from skimage.exposure import equalize_adapthist

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from torch.utils.data import WeightedRandomSampler

import data.pre_cut_dataset as pre_cut_dataset
import utils

import os

def LesionDatasetISIC(subset, **kwargs):
  return ImageDataset(subset=subset, dataset_folder='isic', **kwargs)

def LesionDatasetDermQuest(subset, **kwargs):
  return ImageDataset(subset=subset, dataset_folder='dermquest', **kwargs)

def LesionDatasetDermis(subset, **kwargs):
  return ImageDataset(subset=subset, dataset_folder='dermis', **kwargs)

def StratifiedSampler(dataset):
  ita_df = pd.read_csv('data/lesion/dominant_colors.csv')
  ita_df['image'] = ita_df['image'].str.replace('.jpg', '')
  ita_df.set_index('image', inplace=True)

  # sort the weights by the order of the dataset
  dataset_file_names = dataset._get_files(['train', 'valid', 'test'])
  dataset_file_names = [p.basename(f).replace('.png', '') for f in dataset_file_names]
  ita_angle = np.array([ita_df.loc[f]['ita_angle'] for f in dataset_file_names])

  bins = 6
  weights = np.array([1, 1, 0, 0, 0, 0])
  weights = weights / np.sum(weights)
  hist = np.histogram(ita_angle, bins=bins, density=True)
  #weights = hist[0] / np.sum(hist[0])
  #weights = 1 / weights
  #weights = weights / np.sum(weights)

  print('Stratified sampling:')
  print('Weights:', weights)

  bin_per_image = np.digitize(ita_angle, hist[1], right=True)
  sample_weights = np.zeros(len(dataset))
  for i in range(len(dataset)):
    sample_weights[i] = weights[bin_per_image[i] - 1]
  
  return WeightedRandomSampler(sample_weights, len(sample_weights) * 2, replacement=True)

def get_ita_angle(color_rgb):
  color_lab = color.rgb2lab(np.array([[color_rgb]]))[0][0]
  return np.arctan((color_lab[0] - 50) / color_lab[2]) * 180 / np.pi

def equalize_hist(img, white_point_color):
    #white_point_color_norm = white_point_color / np.linalg.norm(white_point_color)
    #white_point_color_norm *= 255
    white_point_color = cv.cvtColor(np.uint8([[white_point_color]]), cv.COLOR_RGB2LAB)[0][0]
    white_l = white_point_color[0]
    img_lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    img_lab_l = img_lab[:,:,0]
    hist, bins = np.histogram(img_lab_l.flatten(), 256, (0, 256))
    # threshold
    hist[white_l:] = 0
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img_lab_l = cdf[img_lab_l]
    img_lab[:,:,0] = img_lab_l
    img = cv.cvtColor(img_lab, cv.COLOR_LAB2RGB)
    return img

def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    source: https://www.kaggle.com/code/apacheco/shades-of-gray-color-constancy
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    # gamma correct rgb_vec
    rgb_vec = np.power(rgb_vec / 255., 1/1.3) * 255.
    color = rgb_vec.copy()
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    
    return img, rgb_vec, color

def illuminant_from_color(color_rgb):
    rgb_vec = color_rgb.copy()
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/(rgb_norm + 1e-6)
    rgb_vec = 1/(rgb_vec*np.sqrt(3)+1e-6)
    return rgb_vec

def _float_inputs(lab1, lab2, allow_float32=True):
    lab1 = np.asarray(lab1)
    lab2 = np.asarray(lab2)
    lab1 = lab1.astype(np.float32, copy=False)
    lab2 = lab2.astype(np.float32, copy=False)
    return lab1, lab2

def deltaE_cie76(lab1, lab2, channel_axis=-1):
    """
    Based on: https://github.com/scikit-image/scikit-image/blob/main/skimage/color/delta_e.py
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]
    def distance(x, y):
        return np.sqrt((x - y) ** 2)
    
    dist = np.array([distance(L1, L2), distance(a1, a2), distance(b1, b2)])
    return dist.transpose(1, 2, 0)


def dist_img(img, skin_color):
    img_hsv = color.rgb2hsv(img)
    black_regions = np.logical_and(img_hsv[:, :, 1] < 0.1, img_hsv[:, :, 2] < 0.1)
    img_ = img.copy()
    img_[black_regions] = skin_color
    img_lab = color.rgb2lab(img_)
    skin_color_lab = color.rgb2lab(skin_color)
    dist = deltaE_cie76(img_lab, np.ones_like(img_lab) * skin_color_lab) # TODO: Use per-channel distance
    radius = img.shape[0] // 2
    center = (img.shape[0] // 2, img.shape[1] // 2)
    circle_mask = cv.circle(np.zeros_like(img), center, radius, (1, 1, 1), -1)
    dist_masked = dist * circle_mask
    # normalize each channel separately
    for i in range(3):
        max_dist = np.max(dist_masked[:, :, i])
        dist_c = dist[:, :, i]
        dist_c[dist_c > max_dist] = max_dist
        dist[:, :, i] = dist_c / max_dist
        dist[:, :, i] = equalize_adapthist(dist[:, :, i])
    return dist


class ImageDataset(torch.utils.data.Dataset):
  """
  A dataset for RGB images.

  Attributes:
    TODO
  """
  def __init__(self, subset, dataset_folder, subjects=None, augment=False, 
               colorspace='dist'):
    self.dataset_folder = dataset_folder
    self.colorspace = colorspace
    self.num_classes = 6
    self.augment = augment

    assert self.colorspace in ['lab', 'rgb', 'dist']

    if subjects is not None:
      self.subset = 'all'

    if subset == 'all':
      directories = ['train', 'valid', 'test']
    else:
      directories = [subset]

    self.file_names = self._get_files(directories)
    if subjects is not None:
      self.file_names = [f for f in self.file_names if self._get_subject_from_file_name(f) in subjects]
    
    self.subject_id_for_idx = [self._get_subject_from_file_name(f) for f in self.file_names]
    self.subjects = subjects if subjects is not None else set(self.subject_id_for_idx)

    self.skin_colors = None
    file_dir = os.path.dirname(os.path.realpath(__file__))
    self.skin_colors_df = pd.read_csv(p.join(file_dir, 'dominant_colors.csv'))
    self.skin_colors_df['image'] = self.skin_colors_df['image'].str.replace('.jpg', '')
    self.skin_colors_df.set_index('image', inplace=True)
    self.skin_colors = [self.skin_colors_df.loc[s][['R', 'G', 'B']].astype(np.uint8).values for s in self.subject_id_for_idx]

    self.ita_angles = [get_ita_angle(c) for c in self.skin_colors]
    self.classes = [self.class_for_ita_angle(ita) for ita in self.ita_angles]

  def skin_color_from_img(self, input):
    # if self.colorspace == 'lab':
    #   input_rgb = cv.cvtColor(input.transpose(1, 2, 0), cv.COLOR_LAB2RGB)
    # else:
    #   input_rgb = input.transpose(1, 2, 0)
    if input.shape[0] == 3:
      input = input.transpose(1, 2, 0)
    _, _, color = shade_of_gray_cc(input, power=1)
    return color

  def skin_color(self, idx):
    input, _ = self.get_item_np(idx)
    return self.skin_color_from_img(input)
  
  def class_for_ita_angle(self, ita_angle):
    """
    Returns the class label for the given ITA angle based on standard ITA ranges:
      <-30: 0 - Dark
      -30 to 10: 1 - Brown
      10 to 28: 2 - Tan
      28 to 41: 3 - Intermediate
      41 to 55: 4 - Light
      >55: 5 - Very light
    """
    range_limits = [-30, 10, 28, 41, 55]
    for i in range(len(range_limits)):
      if ita_angle < range_limits[i]:
        return i
    return len(range_limits)
  
  def random_skin_color(self, stratified=False):
    def b_center(ita):
      # b_center is 0 for ita = +- 90 and 20 for ita = 50
      return 20 * np.cos(ita * np.pi / 180)

    ita_range_for_class = [
      (-90, -30),
      (-30, 10),
      (10, 28),
      (28, 41),
      (41, 55),
      (55, 90)
    ]

    if stratified:
      class_probs = np.zeros(self.num_classes)
      for c in range(self.num_classes):
        sum = np.sum(self.classes == c)
        if sum == 0:
          class_probs[c] = 1
        else:
          class_probs[c] = 1 / sum
      
      class_probs /= np.sum(class_probs)
    else:
      class_probs = np.ones(self.num_classes) / self.num_classes

    class_idx = np.random.choice(self.num_classes, p=class_probs)

    random_ita = np.random.uniform(*ita_range_for_class[class_idx])
    random_b = np.random.normal(b_center(random_ita), 6)
    random_l = random_b * np.tan(random_ita * np.pi / 180) + 50
    random_a = np.random.normal(10, 2)

    random_lab = np.array([random_l, random_a, random_b])

    random_rgb = color.lab2rgb(random_lab.reshape(1, 1, 3)).reshape(3)# + np.random.normal(0, 0.01, 3)
    return random_rgb

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
      #A.RandomGamma(p=0.7, gamma_limit=(80, 180)),
      #A.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
      A.Flip(p=0.4),
      A.ShiftScaleRotate(p=0.4, rotate_limit=90, scale_limit=0.1, shift_limit=0.1, border_mode=cv.BORDER_CONSTANT, value=0, rotate_method='ellipse'),
      A.GridDistortion(p=0.4, border_mode=cv.BORDER_CONSTANT, value=0)
    ])
  
  def __len__(self):
    return len(self.file_names)

  def get_item_np(self, idx, augmentation=None):
    current_file = self.file_names[idx]

    input = cv.imread(current_file.replace('label/', 'input/').replace('.png', '.jpg'))

    if self.colorspace == 'lab':
      input = cv.cvtColor(input, cv.COLOR_BGR2LAB)
    elif self.colorspace == 'rgb':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
    elif self.colorspace == 'dist':
      input = cv.cvtColor(input, cv.COLOR_BGR2RGB)
      if False:#augmentation is not None:
        input = A.ColorJitter(p=1, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, 
                              always_apply=True)(image=input)['image']
        plt.imshow(input)
        plt.show()
        skin_color = self.skin_color_from_img(input)
        skin_color += np.random.normal(0, 3, 3)
      else:
        if self.skin_colors is None:
          skin_color = self.skin_color_from_img(input)
        else:
          skin_color = self.skin_colors[idx]

        if augmentation is not None:
          aug = np.random.normal(0, 10, 3)
          aug = np.round(aug).astype(int)
          skin_color = skin_color.astype(int) + aug
          skin_color = skin_color.astype(np.uint8)

      input = dist_img(input, skin_color)
    
    input = input.transpose(2, 0, 1)

    mask = cv.imread(current_file, cv.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    if augmentation is not None:
      input = input.transpose(1, 2, 0)
      transformed = augmentation(image=input, mask=mask)
      input = transformed['image']
      input = input.transpose(2, 0, 1)
      mask = transformed['mask']

    mask = mask.astype(np.float32)
    return input, mask

  def __getitem__(self, idx):
    input, label = self.get_item_np(idx, augmentation=self.get_train_augmentation() if self.augment else None)
    to_tensor = ToTensorV2()

    if self.colorspace == 'lab':
      input_rgb = cv.cvtColor(input.transpose(1, 2, 0), cv.COLOR_LAB2RGB)
    else:
      input_rgb = input.transpose(1, 2, 0)

    # input_cc, _, _ = shade_of_gray_cc(input_rgb, power=6)

    # if self.augment:
    #   # tint image with random color
    #   skin_color = self.random_skin_color(stratified=True)
    #   t = illuminant_from_color(skin_color)
    #   input_cc /= t
    #   input_cc = np.clip(input_cc, 0, 255)
    #   input_cc = input_cc.astype(np.uint8)

    #input_cc = equalize_hist(input_cc, self.dominant_color_for_idx[idx])
    #if self.colorspace == 'lab':
      #input_cc = cv.cvtColor(input_cc, cv.COLOR_RGB2LAB)
    #input_cc = input_cc.transpose(2, 0, 1)

    if self.colorspace == 'lab' or self.colorspace == 'rgb':
      input = input.astype(np.float32)
      input = input / 255.

    #input_cc = input_cc.astype(np.float)
    #input_cc = input_cc / 255.

    #input = np.concatenate((input, input_cc), axis=0)
    
    input_tensor, label_tensor = to_tensor(image=input.transpose(1, 2, 0), mask=label).values()
    input_tensor = input_tensor.float()
    label_tensor = label_tensor.unsqueeze(0).float()

    #class_label = self.classes[idx]
    class_label_tensor = torch.zeros(self.num_classes)
    #class_label_tensor[class_label] = 1

    #plt.imshow(input.transpose(1, 2, 0))
    #plt.show()
    #plt.imshow(input.transpose(1, 2, 0)[:,:,[3,4,5]])
    #plt.show()

    return input_tensor, {'seg': label_tensor, 'aux': class_label_tensor}
