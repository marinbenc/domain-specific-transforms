import os
from os import makedirs
import os.path as p
import json

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import cv2 as cv

from medpy.metric.binary import precision as mp_precision
from medpy.metric.binary import recall as mp_recall
from medpy.metric.binary import dc

import torchvision.transforms.functional as F
import torch.nn.functional as nnF
import kornia as K

import skimage.transform as skt

import PIL
from PIL import Image

device = 'cuda'
best_loss = float('inf')

def dull_razor(img):
  """
  Applies the DullRazor algorithm to the image.
  img should be an RGB numpy array of shape (H, W, C) between 0 and 255.
  """
  grayscale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  kernel = cv.getStructuringElement(1, (3,3))
  blackhat = cv.morphologyEx(grayscale, cv.MORPH_BLACKHAT, kernel)
  blurred = cv.GaussianBlur(blackhat, (3,3), cv.BORDER_DEFAULT)
  _, hair_mask = cv.threshold(blurred, 10, 255, cv.THRESH_BINARY)
  result = cv.inpaint(img, hair_mask, 6, cv.INPAINT_TELEA)
  #show_images_row(imgs=[img, hair_mask, result], titles=['Original', 'Hair Mask', 'Result'], figsize=(10, 5))
  return result


# TODO: Move this to separate transforms file
# Idea: Both STN and ITN transforms live in the dataset, don't have separate datasets for each
def itn_transform_lesion(img):
  img += 0.5
  img = img.unsqueeze(0)

  #img = K.enhance.equalize(img)
  img = K.enhance.equalize_clahe(img, clip_limit=2.)
  img = K.filters.UnsharpMask((9,9), (4,4))(img)

  img_np = img.squeeze(0).numpy().transpose(1, 2, 0)
  img_np = dull_razor((img_np * 255).astype(np.uint8)).astype(np.float32) / 255.
  
  img = torch.from_numpy(img_np).permute(2, 0, 1)
  img -= 0.5

  return img

def get_affine_from_bbox(x, y, w, h, size):
  """
  Returns an affine transformation matrix in OpenCV-expected format that
  will crop the image to the bounding box.
  """
  scale_x = size / w
  scale_y = size / h
  M = np.array([[scale_x, 0, -x * scale_x], [0, scale_y, -y * scale_y]])
  return M

def get_theta_from_bbox(x, y, w, h, size):
  """
  Returns an affine transformation matrix in PyTorch-expected format that 
  will crop the image to the bounding box.
  """
  scale_x = size / w
  scale_y = size / h

  x_t = (x + w / 2) / size * 2 - 1
  y_t = (y + h / 2) / size * 2 - 1

  theta = np.array([[1 / scale_x, 0, x_t], [0, 1 / scale_y, y_t]], dtype=np.float32)
  return theta

def transform_keypoints(kps, meta, invert=False):
    keypoints = kps.copy()
    if invert:
        meta = np.linalg.inv(meta)
    keypoints[:, :2] = np.dot(keypoints[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return keypoints

def crop_to_label(input, label, padding=32, bbox_aug=0):
  """ 
  Crop input to bbox enclosing label, with padding and bbox augmentation.
  Args:
    input: input image
    label: label image
    padding: padding around label
    bbox_aug: random bbox augmentation in pixels, 
              each bbox parameter (x, y, w, h) is augmented 
              by a random value in [-bbox_aug, bbox_aug]
  """
  original_size = label.shape[:2]

  label_th = label.copy()
  label_th[label_th > 0.5] = 1
  label_th[label_th <= 0.5] = 0
  label_th = label_th.astype(np.uint8)
  bbox = cv.boundingRect(label_th)

  x, y, w, h = bbox

  if bbox_aug > 0:
    augs = np.random.randint(-bbox_aug, bbox_aug, size=4)
    x += augs[0]
    y += augs[1]
    w += augs[2]
    h += augs[3]

  x -= padding
  y -= padding
  w += 2 * padding
  h += 2 * padding

  theta = get_theta_from_bbox(x, y, w, h, size=original_size[0])
  M = get_affine_from_bbox(x, y, w, h, size=original_size[0])

  input_cropped = cv.warpAffine(input, M, original_size, flags=cv.INTER_LINEAR)
  label_cropped = cv.warpAffine(label, M, original_size, flags=cv.INTER_NEAREST)

  return input_cropped, label_cropped, theta

def save_args(args, folder):
    args_file = os.path.join(os.path.dirname(__file__), 'runs', args.log_name, folder, 'args.json')
    makedirs(os.path.join(os.path.dirname(__file__), 'runs', args.log_name, folder), exist_ok=True)
    with open(args_file, 'w') as fp:
        json.dump(vars(args), fp)

def save_checkpoint(name, log_dir, model, epoch, optimizer, loss):
    file_name = p.join(log_dir, name)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }, file_name)

# TODO: Move to separate file
class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, log_dir, checkpoint_name, scheduler=None, device='cuda'):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.log_dir = log_dir
    self.checkpoint_name = checkpoint_name
    self.scheduler = scheduler
    self.device = device

    self.writer = SummaryWriter(log_dir=self.log_dir)
    self.best_loss = float('inf')
    self.epochs_since_best = 0

  def _to_device(self, data):
    if isinstance(data, (list, tuple)):
      if len(data) == 1:
        return self._to_device(data[0])
      return [self._to_device(x) for x in data]
    elif isinstance(data, dict):
      return {k: self._to_device(v) for k, v in data.items()}
    else:
      return data.to(self.device, non_blocking=True)

  def train(self, epochs):
    self.epochs_since_best = 0
    self.best_loss = float('inf')

    for epoch in range(epochs):
      early_stop = self._train_epoch(epoch)
      if early_stop:
        break

  def _plot_grad_flow(self, named_parameters):
      '''Plots the gradients flowing through different layers in the net during training.
      Can be used for checking for possible gradient vanishing / exploding problems.
      
      Usage: Plug this function in Trainer class after loss.backwards() as 
      "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
      ave_grads = []
      max_grads= []
      layers = []
      for n, p in named_parameters:
          if(p.requires_grad) and ("bias" not in n) and (p.grad is not None):
              layers.append(n)
              ave_grads.append(p.grad.abs().mean().cpu())
              max_grads.append(p.grad.abs().max().cpu())
      plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
      plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
      plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
      plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
      plt.xlim(left=0, right=len(ave_grads))
      plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
      plt.xlabel("Layers")
      plt.ylabel("average gradient")
      plt.title("Gradient flow")
      plt.grid(True)
      plt.legend([Line2D([0], [0], color="c", lw=4),
                  Line2D([0], [0], color="b", lw=4),
                  Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
      plt.tight_layout()
      plt.show()

  def _train_epoch(self, epoch):
    """
    Returns:
      early_stop: True if early stopping should be performed
    """
    if epoch == 0:
      self.best_loss = float('inf')
    
    self.model.train()
    loss_total = 0
    for batch_idx, batch in enumerate(self.train_loader):
      input = self._to_device(self.get_input(batch))
      target = self._to_device(self.get_target(batch))

      #show_torch(imgs=[data[0] + 0.5, target[0]])
      self.optimizer.zero_grad()
      output = self.model(input)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      loss_total += loss.item()
    
    #self._plot_grad_flow(self.model.named_parameters())
    loss_total /= len(self.train_loader)
    self.writer.add_scalar('Loss/train', loss_total, epoch)

    print(f'Train Epoch: {epoch}\tTrain Loss: {loss_total:.6f}', end='', flush=True)

    if self.val_loader is not None:
      loss_total = 0
      self.model.eval()
      with torch.no_grad():
        for batch in self.val_loader:
          input = self._to_device(self.get_input(batch))
          target = self._to_device(self.get_target(batch))
          output = self.model(input)
          loss = self.loss_fn(output, target)
          loss_total += loss.item()
      loss_total /= len(self.val_loader)
      self.writer.add_scalar('Loss/valid', loss_total, epoch)
      print(f'\tValid Loss: {loss_total:.6f}', end='', flush=True)
    
    print()

    if self.scheduler is not None:
      self.scheduler.step(loss_total)

    if loss_total < self.best_loss and True:
        print('Saving new best model...')
        self.best_loss = loss_total
        self.epochs_since_best = 0
        save_checkpoint(self.checkpoint_name, self.writer.log_dir, self.model, epoch, self.optimizer, loss_total)
    
    if self.epochs_since_best > 10:
      print('Early stopping')
      return True
    
    # if (epoch) % 20 == 0:
    #   show_torch(imgs=[input[0][0], output['img_stn'][0], target['img_stn'][0]])

    self.epochs_since_best += 1
    return False

  def get_input(self, batch):
    """Convert data loader output to input of the model"""
    return batch[0]

  def get_target(self, batch):
    """Convert data loader output to target of the model"""
    return batch[1:]

def _thresh(img):
  img[img > 0.5] = 1
  img[img <= 0.5] = 0
  return img

def dsc(y_pred, y_true):
  y_pred = _thresh(y_pred)
  y_true = _thresh(y_true)

  if not np.any(y_true):
    return 0 if np.any(y_pred) else 1

  score = dc(y_pred, y_true)
  return score

def iou(y_pred, y_true):
  y_pred = _thresh(y_pred)
  y_true = _thresh(y_true)

  intersection = np.logical_and(y_pred, y_true)
  union = np.logical_or(y_pred, y_true)
  if not np.any(union):
    return 0 if np.any(y_pred) else 1
  
  return intersection.sum() / float(union.sum())

def precision(y_pred, y_true):
  #y_pred = _thresh(y_pred).astype(np.int)
  #y_true = _thresh(y_true).astype(np.int)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, precision is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.
  
  return mp_precision(y_pred, y_true)

def recall(y_pred, y_true):
  y_pred = _thresh(y_pred).astype(np.int)
  y_true = _thresh(y_true).astype(np.int)

  if y_true.sum() <= 5:
    # when the example is nearly empty, avoid division by 0
    # if the prediction is also empty, recall is 1
    # otherwise it's 0
    return 1 if y_pred.sum() <= 5 else 0

  if y_pred.sum() <= 5:
    return 0.
  
  r = mp_recall(y_pred, y_true)
  return r

def listdir(path):
  """ List files but remove hidden files from list """
  return [item for item in os.listdir(path) if item[0] != '.']

def show_torch(imgs, titles=None, show=True, save=False, save_path=None, figsize=(6.4, 4.8), **kwargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), **kwargs)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if titles is not None:
          axs[0, i].set_title(titles[i])
    if save:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()



def show_images_row(imgs, titles=None, rows=1, figsize=(6.4, 4.8), show=True, **kwargs):
  '''
      Display grid of cv2 images
      :param img: list [cv::mat]
      :param title: titles
      :return: None
  '''
  assert ((titles is None) or (len(imgs) == len(titles)))
  num_images = len(imgs)

  fig = plt.figure(figsize=figsize)
  for n, image in enumerate(imgs):
      ax = fig.add_subplot(rows, int(np.ceil(num_images / float(rows))), n + 1)
      plt.imshow(image, **kwargs)
      plt.axis('off')
  
  if show:
    plt.show()