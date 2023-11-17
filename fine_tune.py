import utils

import random
import os
import shutil
import os.path as p
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import segmentation_models_pytorch as smp

from torch.utils.tensorboard import SummaryWriter

import argparse
import datetime
import numpy as np

import data.datasets as data
import seg.loss as loss
import seg.train_seg as seg
import stn.train_stn as stn
import model as m
import stn.stn_losses as losses

from torch.nn import BCELoss
import kornia as K

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

device = 'cuda'

def get_model(dataset, log_name):
  stn_model = stn.get_model(dataset)
  stn_checkpoint_f = p.join('runs', log_name, 'stn', 'stn_best.pth')
  stn_checkpoint = torch.load(stn_checkpoint_f)
  stn_model.load_state_dict(stn_checkpoint['model'])
#   for param in stn_model.parameters():
#     param.requires_grad = False

  # TODO: Check if loss function is the one shifting the STN

  seg_model = seg.get_model(dataset)
  seg_checkpoint_f = p.join('runs', log_name, 'seg', 'seg_best.pth')
  seg_checkpoint = torch.load(seg_checkpoint_f)
  seg_model.load_state_dict(seg_checkpoint['model'])

  model = m.TransformedSegmentation(stn_model, seg_model)
  return model

    
def fine_tune(batch_size, epochs, lr, dataset, subset, log_name):
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)

    os.makedirs(log_dir/'fine', exist_ok=True)


    train_dataset, val_dataset = data.get_datasets(dataset, subset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
    val_loader = DataLoader(val_dataset, worker_init_fn=worker_init)

    model = get_model(train_dataset, log_name)
    # Output the STN theta to calculate smoothness loss
    model.output_theta = True

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    loss_fn = loss.DiceLoss()
    smoothness = losses.IdentityTransformLoss()

    def calculate_loss(output, target):
        output_loc_img, output_img, ouput_theta = output
        #smoothness_loss = smoothness(ouput_theta)
        loc_img_loss = loss_fn(output_loc_img, target)
        img_loss = loss_fn(output_img, target)
        return img_loss + loc_img_loss #smoothness_loss * 0.1

    writer = SummaryWriter(log_dir=f'{log_dir}/fine')
    for epoch in range(1, epochs + 1):
      utils.train(model, calculate_loss, optimizer, epoch, train_loader, val_loader, writer=writer, checkpoint_name='fine_best.pth', scheduler=scheduler)
    writer.close()

#TODO: Save arguments json file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine tuning'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--dataset', type=str, choices=data.dataset_choices, default='lesion', help='which dataset to use'
    )
    parser.add_argument(
        '--subset', type=str, choices=data.lesion_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--log-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), help='name of folder where checkpoints are stored',
    )
    args = parser.parse_args()
    log_dir = Path(f'runs/{args.log_name}')
    if p.exists(log_dir/'fine'):
        shutil.rmtree(log_dir/'fine')

    utils.save_args(args, 'fine')
    fine_tune(**vars(args))