import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()
    self.smooth = 1.0
    self.iters = 0

  @staticmethod
  def abs_exp_loss(y_pred, y_true, pow):
    return torch.abs((y_pred - y_true) ** pow).mean()

  def forward(self, y_pred, y_true):
    if isinstance(y_pred, dict):
      y_pred = y_pred['seg']
    if isinstance(y_true, dict):
      y_true = y_true['seg']

    dscs = torch.zeros(y_pred.shape[1])

    # visualize
    utils.show_torch(imgs=[y_pred[0].squeeze(), y_true[0].squeeze()], titles=['pred', 'true'])

    for i in range(y_pred.shape[1]):
      y_pred_ch = y_pred[:, i].contiguous().view(-1)
      y_true_ch = y_true[:, i].contiguous().view(-1)
      intersection = (y_pred_ch * y_true_ch).sum()
      dscs[i] = (2. * intersection + self.smooth) / (
          y_pred_ch.sum() + y_true_ch.sum() + self.smooth
      )

    return (1. - torch.mean(dscs))

class TabeTrainer(utils.Trainer):
  def __init__(self, model, train_loader, val_loader, seg_loss, aux_loss, device, lr, momentum, alpha, 
               log_dir, checkpoint_name):
    super().__init__(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=None, loss_fn=None, 
                     log_dir=log_dir, checkpoint_name=checkpoint_name, device=device)

    self.seg_loss = seg_loss
    self.aux_loss = aux_loss
    self.alpha = alpha
    self.viz_freq = -1

    segmentation_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    segmentation_parameters = list(filter(lambda p: p.requires_grad, segmentation_parameters))
    self.optimizer_seg = torch.optim.SGD(segmentation_parameters, lr=lr, momentum=momentum)
    self.optimizer_aux = torch.optim.SGD(self.model.aux_head.parameters(), lr=lr, momentum=momentum)
    self.optimizer_confusion = torch.optim.SGD(self.model.encoder.parameters(), lr=lr, momentum=momentum)

    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_seg, mode='min', factor=0.01, patience=3, verbose=True)

  def confusion_loss(self, output, target, alpha):
    uniform_distribution = torch.FloatTensor(output.shape).uniform_(0, 1).to(self.device)
    loss = -alpha * (torch.sum(uniform_distribution * torch.log(output + 1e-8))) / float(output.shape[0])
    return loss

  def _train_epoch(self, epoch):
    """
    Returns:
      early_stop: True if early stopping should be performed
    """
    if epoch == 0:
      self.best_loss = float('inf')

    optims = [self.optimizer_seg, self.optimizer_aux, self.optimizer_confusion]
    models = [self.model.segmentation_head, self.model.aux_head, self.model.encoder]

    for m in models:
      m.train()

    loss_names = ['loss_seg', 'loss_aux', 'loss_conf']
    loss_totals = { n: 0 for n in loss_names }

    for batch_idx, batch in enumerate(self.train_loader):
      input = self._to_device(self.get_input(batch))
      target = self._to_device(self.get_target(batch))
      target_seg = target['seg']
      target_aux = target['aux']

      for o in optims:
        o.zero_grad()

      # Main segmentation

      feat_out = self.model.encoder(input)
      seg_out = self.model.decoder(*feat_out)
      seg_out = self.model.segmentation_head(seg_out)

      if self.viz_freq != -1 and epoch % self.viz_freq == 0 and batch_idx == 0:
        utils.show_torch(imgs=[input[0], seg_out[0], target_seg[0]], titles=['input', 'pred', 'true'])

      loss_seg = self.seg_loss(seg_out, target_seg)
      loss_totals['loss_seg'] += loss_seg.item()

      # Auxiliary confusion loss

      _, aux_out_p = self.model.aux_head(feat_out[-1])
      loss_conf = self.confusion_loss(aux_out_p, target_aux, self.alpha)
      loss_totals['loss_conf'] += loss_conf.item()

      loss = loss_seg + loss_conf
      loss.backward()
      self.optimizer_seg.step()
      self.optimizer_confusion.step()

      # Auxiliary classification loss

      self.optimizer_seg.zero_grad()
      self.optimizer_aux.zero_grad()

      feat_out = self.model.encoder(input)
      aux_out_logits, _ = self.model.aux_head(feat_out[-1])
      aux_loss = self.aux_loss(aux_out_logits, target_aux)
      loss_totals['loss_aux'] += aux_loss.item()

      aux_loss.backward()
      self.optimizer_seg.step()
      self.optimizer_aux.step()

      #show_torch(imgs=[input[0], target['seg'][0]])

    loss_totals = { n: v / len(self.train_loader) for n, v in loss_totals.items() }
    for n, v in loss_totals.items():
      self.writer.add_scalar(f'Loss/train/{n}', v, epoch)

    #self._plot_grad_flow(self.model.named_parameters())

    loss_string = ' '.join([f'{n}: {v:.6f}' for n, v in loss_totals.items()])
    print(f'Train Epoch: {epoch}\tTrain: {loss_string}', end='', flush=True)

    if self.val_loader is not None:
      loss_total = 0
      for m in models:
        m.eval()
      with torch.no_grad():
        for batch in self.val_loader:
          input = self._to_device(self.get_input(batch))
          target = self._to_device(self.get_target(batch))
          target_seg = target['seg']
          out_seg = self.model.forward(input)
          loss = self.seg_loss(out_seg, target_seg)
          loss_total += loss.item()

        if epoch % self.viz_freq == 0:
          self.writer.add_image('Valid/seg_out', out_seg[0].detach().cpu().numpy(), epoch)
          self.writer.add_image('Valid/target_seg', target_seg[0].detach().cpu().numpy(), epoch)
      loss_total /= len(self.val_loader)
      self.writer.add_scalar('Loss/valid', loss_total, epoch)
      print(f'\tValid Loss: {loss_total:.6f}', end='', flush=True)
    
    print()

    if self.scheduler is not None:
      self.scheduler.step(loss_total)
      self.optimizer_aux.param_groups[0]['lr'] = self.optimizer_seg.param_groups[0]['lr']
      self.optimizer_confusion.param_groups[0]['lr'] = self.optimizer_seg.param_groups[0]['lr']

    if loss_total < self.best_loss and True:
        print('Saving new best model...')
        self.best_loss = loss_total
        self.epochs_since_best = 0
        utils.save_checkpoint(self.checkpoint_name, self.writer.log_dir, self.model, epoch, self.optimizer, loss_total)
    
    if self.epochs_since_best > 10:
      print('Early stopping')
      return True
    
    self.epochs_since_best += 1
    return False

class AuxHead(nn.Module):
  def __init__(self, n_classes):
    super().__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.head = nn.Sequential(
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, n_classes),
    )
    self.activation = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.pool(x).squeeze()
    x = self.head(x)
    return x, self.activation(x)

class TabeUNet():
  def __init__(self, base_model, n_aux_classes):
    self.n_aux_classes = n_aux_classes
    self.encoder = base_model.encoder
    self.decoder = base_model.decoder
    encoder_layers = list(base_model.encoder.children())
    self.segmentation_head = base_model.segmentation_head
    self.aux_head = AuxHead(n_classes=n_aux_classes)
  
  def forward(self, x):
    features = self.encoder(x)
    decoder_output = self.decoder(*features)
    masks = self.segmentation_head(decoder_output)
    return masks

  def state_dict(self):
    return {
      'n_aux_classes': self.n_aux_classes, 
      'encoder': self.encoder.state_dict(),
      'decoder': self.decoder.state_dict(),
      'segmentation_head': self.segmentation_head.state_dict(),
      'aux_head': self.aux_head.state_dict(),
    }
  
  def load_state_dict(self, state_dict):
    self.n_aux_classes = state_dict['n_aux_classes']
    self.encoder.load_state_dict(state_dict['encoder'])
    self.decoder.load_state_dict(state_dict['decoder'])
    self.segmentation_head.load_state_dict(state_dict['segmentation_head'])
    self.aux_head.load_state_dict(state_dict['aux_head'])

  def eval(self):
    self.encoder.eval()
    self.decoder.eval()
    self.segmentation_head.eval()
    self.aux_head.eval()
  
  def train(self):
    self.encoder.train()
    self.decoder.train()
    self.segmentation_head.train()
    self.aux_head.train()
