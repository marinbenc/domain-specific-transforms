import os
import os.path as p

import torch

import matplotlib.pyplot as plt
import numpy as np

from medpy.metric.binary import precision as mp_precision
from medpy.metric.binary import recall as mp_recall
from medpy.metric.binary import dc

import torchvision.transforms.functional as F

device = 'cuda'
best_loss = float('inf')

def save_checkpoint(name, log_dir, model, epoch, optimizer, loss):
    file_name = p.join(log_dir, name)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }, file_name)

def train(model, loss_fn, optimizer, epoch, train_loader, val_loader, writer, checkpoint_name):
    global best_loss
    if epoch == 0:
      best_loss = float('inf')
    
    model.train()
    loss_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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
      loss_total /= len(train_loader)
      writer.add_scalar('Loss/valid', loss_total, epoch)
      print(f'\tValid Loss: {loss_total:.6f}', end='', flush=True)
    
    print()

    if loss_total < best_loss and True:
        print('Saving new best model...')
        best_loss = loss_total
        save_checkpoint(checkpoint_name, writer.log_dir, model, epoch, optimizer, loss_total)

    #utils.show_torch(imgs=[data[0] + 0.5, output[0] + 0.5, target[0] + 0.5])

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