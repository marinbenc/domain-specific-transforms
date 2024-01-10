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

def get_model(dataset, num_channels=-1):
    if num_channels == -1:
        num_channels = dataset.in_channels
    model = smp.Unet('resnet34', in_channels=num_channels, classes=1, activation='sigmoid')
    model.to('cuda')
    return model

def train_seg(batch_size, epochs, lr, dataset, subset, log_name, folds):
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)

    os.makedirs(f'runs/{log_name}/seg', exist_ok=True)

    datasets = data.get_kfolds_datasets(dataset, subset, folds, log_name)

    for fold, (train_dataset, val_dataset) in enumerate(datasets):
        print(f'Fold {fold}')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)
        val_loader = DataLoader(val_dataset, worker_init_fn=worker_init)

        model = get_model(train_dataset)

        loss_fn = loss.DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        os.makedirs(f'runs/{log_name}/seg/fold_{fold}', exist_ok=True)

        writer = SummaryWriter(log_dir=f'runs/{log_name}/seg/fold_{fold}')
        for epoch in range(epochs):
            early_stop = utils.train(model, loss_fn, optimizer, epoch, train_loader, val_loader, writer=writer, checkpoint_name='seg_best.pth', scheduler=scheduler)
            if early_stop:
                break
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
    parser.add_argument(
        '--folds', type=int, default=5, help='number of folds to use for cross validation',
    )
    args = parser.parse_args()
    log_dir = Path(f'runs/{args.log_name}')
    if p.exists(log_dir/'seg'):
        shutil.rmtree(log_dir/'seg')

    utils.save_args(args, 'seg')
    train_seg(**vars(args))