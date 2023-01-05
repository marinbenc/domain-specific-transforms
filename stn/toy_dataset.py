import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import cv2

class ToyDataset(Dataset):

  def __init__(self):
    label_size = 100
    self.label_np = np.zeros((label_size, label_size))
    self.label_np[20:80, 40:60] = 1
    self.label = torch.from_numpy(self.label_np).unsqueeze(0).float()
    self.transform = A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(-0.5, 0.1), rotate_limit=90, p=1, border_mode=cv2.BORDER_CONSTANT)

  def __len__(self):
    return 100

  def __getitem__(self, idx):
    image = self.transform(image=self.label_np)['image']
    image = torch.from_numpy(image).unsqueeze(0).float()
    return image, self.label
