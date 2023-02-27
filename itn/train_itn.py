import utils

import random
import os
import os.path as p

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import argparse
import datetime
import numpy as np
import shutil

import data.datasets as data
import itn.itn_dataset as itn_dataset
import itn.itn_model as itn_model
import seg.train_seg as seg

import kornia as K

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

best_loss = float('inf')

import torch
import torch.nn as nn
import torch.nn.functional as F

import itn.pix2pix as pix2pix

def get_model(dataset):
    #seg_model = seg.get_model(dataset, sigmoid_activation=False)
    #model = itn_model.ITN(seg_model.encoder)
    model = ITN2D(input_channels=1)
    model.to('cuda')
    return model

def train_itn(batch_size, epochs, lr, dataset, subset, log_name):
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)

    os.makedirs(log_dir, exist_ok=True)

    dataset_class = data.get_dataset_class(dataset)
    train_dataset = itn_dataset.ITNDataset(dataset_class, subset=subset, directory='train', augment=False)
    valid_dataset = itn_dataset.ITNDataset(dataset_class, subset=subset, directory='valid', augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
    valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init)
    
    #model = get_model(train_dataset)

    # TODO: Check if you should pretrain with STN or SEG?
    # seg_path = p.join(log_dir, '../seg', 'seg_best.pth')
    # if p.exists(seg_path):
    #     print('Transfer learning with: ' + seg_path)
    #     saved_seg = torch.load(p.join(log_dir, '../seg', 'seg_best.pth'))
    #     encoder_dict = {key.replace('encoder.', ''): value 
    #             for (key, value) in saved_seg['model'].items()
    #             if 'encoder.' in key}
    #     # pretrain with the SEG model
    #     model.encoder.load_state_dict(encoder_dict)
    # else:
    #     print('No saved SEG model exists, skipping transfer learning...')

    generator = pix2pix.Generator()
    generator.weight_init(mean=0.0, std=0.02)
    generator.to('cuda')

    discriminator = pix2pix.Discriminator()
    discriminator.weight_init(mean=0.0, std=0.02)
    discriminator.to('cuda')

    generator.train()
    discriminator.train()

    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True, min_lr=1e-15, eps=1e-15)

    writer = SummaryWriter(log_dir=log_dir)

    #loss = torch.nn.MSELoss()
    loss = torch.nn.L1Loss()

    for epoch in range(1, epochs + 1):
      d_loss_total = 0
      g_loss_total = 0

      for (i, (x, y)) in enumerate(train_loader):
        # discriminator training
        x = x.to('cuda')
        y = y.to('cuda')

        discriminator.zero_grad()
        d_result = discriminator(x, y).squeeze()
        d_real_loss = F.binary_cross_entropy_with_logits(d_result, torch.ones_like(d_result))

        g_result = generator(x)
        d_result = discriminator(x, g_result).squeeze()
        d_fake_loss = F.binary_cross_entropy_with_logits(d_result, torch.zeros_like(d_result))

        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss_total += d_loss.item()
        d_loss.backward()
        d_optim.step()

        # generator training
        generator.zero_grad()
        g_result = generator(x)
        d_result = discriminator(x, g_result).squeeze()
        g_loss = F.binary_cross_entropy_with_logits(d_result, torch.ones_like(d_result)) + 100 * loss(g_result, y)
        g_loss_total += g_loss.item()
        g_loss.backward()
        g_optim.step()

      g_loss_total /= len(train_loader)
      d_loss_total /= len(train_loader)
      writer.add_scalar('Loss/generator', g_loss_total, epoch)
      writer.add_scalar('Loss/discriminator', d_loss_total, epoch)

      print('Epoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch, g_loss_total, d_loss_total))

      if epoch % 100 == 0:
        utils.show_torch(imgs=[x[0], y[0], g_result[0]])

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training'
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
        '--subset', type=str, choices=data.all_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--log-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), help='name of folder where checkpoints are stored',
    )

    args = parser.parse_args()
    log_dir = f'runs/{args.log_name}/itn'
    if p.exists(log_dir):
        shutil.rmtree(log_dir)

    utils.save_args(args, 'itn')
    train_itn(**vars(args))