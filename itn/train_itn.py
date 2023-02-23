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

class ITN2D(nn.Module):

    def __init__(self, input_channels):
        super(ITN2D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(input_channels, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv2d(4, 8, kernel_size=2, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, bias=use_bias)
        self.conv22 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, bias=use_bias)
        self.conv13 = nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=use_bias)
        self.conv15 = nn.Conv2d(2, input_channels, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x2 = self.up2(x3) + x2
        x2 = F.relu(self.conv22(x2))
        x1 = self.up1(x2) + x1
        x1 = F.relu(self.conv13(x1))
        x1 = F.relu(self.conv14(x1))
        x = self.conv15(x1)

        return x

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
    train_dataset = itn_dataset.ITNDataset(dataset_class, subset=subset, directory='train')
    valid_dataset = itn_dataset.ITNDataset(dataset_class, subset=subset, directory='valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
    valid_loader = DataLoader(valid_dataset, worker_init_fn=worker_init)
    
    model = get_model(train_dataset)

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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True, min_lr=1e-15, eps=1e-15)

    writer = SummaryWriter(log_dir=log_dir)

    loss = torch.nn.MSELoss()
    #loss = torch.nn.L1Loss()

    for epoch in range(1, epochs + 1):
        utils.train(model, loss, optimizer, epoch, train_loader, valid_loader, writer=writer, checkpoint_name='itn_best.pth', scheduler=scheduler)
    
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