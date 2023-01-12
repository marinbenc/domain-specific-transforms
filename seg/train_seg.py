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
from . import loss

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

device = 'cuda'

def get_model(dataset):
    model = smp.Unet('resnet18', in_channels=dataset.in_channels, classes=1, activation='sigmoid')
    model.to('cuda')
    return model

def train_seg(batch_size, epochs, lr, dataset, subset, log_name):
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)

    log_dir = Path(f'runs/{log_name}')
    if p.exists(log_dir/'seg'):
        shutil.rmtree(log_dir/'seg')
    os.makedirs(log_dir/'seg', exist_ok=True)


    train_dataset, val_dataset = data.get_datasets(dataset, subset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
    val_loader = DataLoader(val_dataset, worker_init_fn=worker_init)

    model = get_model(train_dataset)
    saved_stn = torch.load(p.join(log_dir, 'stn', 'stn_best.pth'))
    encoder_dict = {key.replace('loc_net.', ''): value 
            for (key, value) in saved_stn['model'].items()
            if 'loc_net.' in key}
    # pretrain with the STN model
    model.encoder.load_state_dict(encoder_dict)

    loss_fn = loss.DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=f'{log_dir}/seg')
    for epoch in range(1, epochs + 1):
      utils.train(model, loss_fn, optimizer, epoch, train_loader, val_loader, writer=writer, checkpoint_name='seg_best.pth')
    writer.close()

#TODO: Save arguments json file

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
        '--subset', type=str, choices=data.lesion_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--log-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), help='name of folder where checkpoints are stored',
    )
    args = parser.parse_args()
    train_seg(**vars(args))