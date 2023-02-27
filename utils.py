import os
from os import makedirs
import os.path as p
import json

import torch

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from medpy.metric.binary import precision as mp_precision
from medpy.metric.binary import recall as mp_recall
from medpy.metric.binary import dc

import torchvision.transforms.functional as F
import kornia as K

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

  x = max(0, x - padding)
  y = max(0, y - padding)
  w = min(w + 2 * padding, label_th.shape[1] - x)
  h = min(h + 2 * padding, label_th.shape[0] - y)

  input_cropped = input[y:y+h, x:x+w, :].copy()
  label_cropped = label[y:y+h, x:x+w]

  padding_left = max(0, padding - x)
  padding_right = max(0, padding - (original_size[1] - (x + w)))
  padding_top = max(0, padding - y)
  padding_bottom = max(0, padding - (original_size[0] - (y + h)))

  input_cropped = cv.copyMakeBorder(input_cropped, padding_top, padding_bottom, padding_left, padding_right, cv.BORDER_CONSTANT, value=0)
  label_cropped = cv.copyMakeBorder(label_cropped, padding_top, padding_bottom, padding_left, padding_right, cv.BORDER_CONSTANT, value=0)

  input_cropped = cv.resize(input_cropped, original_size, interpolation=cv.INTER_LINEAR)
  label_cropped = np.array(Image.fromarray(label_cropped).resize(original_size, resample=PIL.Image.NEAREST))
  return input_cropped, label_cropped

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

def train(model, loss_fn, optimizer, epoch, train_loader, val_loader, writer, checkpoint_name, scheduler=None):
    global best_loss
    if epoch == 0:
      best_loss = float('inf')
    
    model.train()
    loss_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #show_torch(imgs=[data[0] + 0.5, target[0]])
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        
    loss_total /= len(train_loader)
    writer.add_scalar('Loss/train', loss_total, epoch)

    print(f'Train Epoch: {epoch}\tTrain Loss: {loss_total:.6f}', end='', flush=True)

    if val_loader is not None:
      loss_total = 0
      model.eval()
      with torch.no_grad():
        for (data, target) in val_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          loss = loss_fn(output, target)
          loss_total += loss.item()
      loss_total /= len(val_loader)
      writer.add_scalar('Loss/valid', loss_total, epoch)
      print(f'\tValid Loss: {loss_total:.6f}', end='', flush=True)
    
    print()

    if scheduler is not None:
      scheduler.step(loss_total)

    if loss_total < best_loss and True:
        print('Saving new best model...')
        best_loss = loss_total
        save_checkpoint(checkpoint_name, writer.log_dir, model, epoch, optimizer, loss_total)

    #if (epoch - 1) % 25 == 0:
    #  show_torch(imgs=[data[0], output[0], target[0]])

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
  y_pred = _thresh(y_pred).astype(np.int)
  y_true = _thresh(y_true).astype(np.int)

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

def show_torch(imgs, show=True, save=False, save_path=None, figsize=(6.4, 4.8), **kwargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), **kwargs)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
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