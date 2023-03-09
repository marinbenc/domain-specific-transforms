import utils

import random
import os
import os.path as p

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

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

def get_model(dataset):
    seg_model = seg.get_model(dataset, sigmoid_activation=False)
    seg_model.train()
    model = itn_model.ITN(seg_model.encoder)
    #model = ITN2D(input_channels=1)
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
    model.output_img = False

    # TODO: Check if you should pretrain with STN or SEG?
    seg_path = p.join(log_dir, '../seg', 'seg_best.pth')
    if p.exists(seg_path):
        print('Transfer learning with: ' + seg_path)
        saved_seg = torch.load(p.join(log_dir, '../seg', 'seg_best.pth'))
        encoder_dict = {key.replace('encoder.', ''): value 
                for (key, value) in saved_seg['model'].items()
                if 'encoder.' in key}
        # pretrain with the SEG model
        model.loc_net.load_state_dict(encoder_dict)
    else:
        print('No saved SEG model exists, skipping transfer learning...')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True, min_lr=1e-15, eps=1e-15)

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    def calculate_loss(output, target):
        target_img, target_thresh = target
        output_img = output['img_stn']
        output_thresh = output['threshold']
        th_loss = mse(output_thresh, target_thresh)
        img_loss = l1(output_img, target_img)
        return th_loss + img_loss

    trainer = utils.Trainer(model, optimizer, calculate_loss, train_loader, valid_loader, log_dir=f'runs/{args.log_name}/itn', checkpoint_name='itn_best.pth', scheduler=scheduler)
    trainer.train(epochs)

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