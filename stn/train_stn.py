from . import stn_model, stn_dataset, stn_losses
import utils

import random
import os
import os.path as p

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import segmentation_models_pytorch as smp

from torch.utils.tensorboard import SummaryWriter
from monai.losses import LocalNormalizedCrossCorrelationLoss, GlobalMutualInformationLoss, BendingEnergyLoss
import stn.stn_losses as losses

import argparse
import datetime
import numpy as np
import shutil

import data.datasets as data
import seg.train_seg as seg

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

best_loss = float('inf')

def get_model(dataset):
    loc_net = seg.get_model(dataset).encoder
    model = stn_model.STN(loc_net=loc_net, output_theta=False)
    model.to('cuda')
    return model

def train_stn(batch_size, epochs, lr, dataset, subset, log_name):
    def worker_init(worker_id):
        np.random.seed(2022 + worker_id)

    log_dir = f'runs/{log_name}/stn'
    if p.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    train_dataset, valid_dataset = data.get_datasets(dataset, subset, augment=False)
    stn_train_dataset = stn_dataset.STNDataset(wrapped_dataset=train_dataset)
    train_loader = DataLoader(stn_train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init)

    stn_valid_dataset = stn_dataset.STNDataset(wrapped_dataset=valid_dataset)
    valid_loader = DataLoader(stn_valid_dataset, worker_init_fn=worker_init)
    
    model = get_model(train_dataset)
    model.output_theta = True

    seg_path = p.join(log_dir, '../seg', 'seg_best.pth')
    if p.exists(seg_path):
        print('Transfer learning with: ' + seg_path)
        saved_seg = torch.load(p.join(log_dir, '../seg', 'seg_best.pth'))
        encoder_dict = {key.replace('encoder.', ''): value 
                for (key, value) in saved_seg['model'].items()
                if 'encoder.' in key}
        # pretrain with the STN model
        model.loc_net.load_state_dict(encoder_dict)
    else:
        print('No saved SEG model exists, skipping transfer learning...')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

    writer = SummaryWriter(log_dir=log_dir)

    loss = torch.nn.L1Loss()
    smoothness = losses.IdentityTransformLoss()

    def calculate_loss(output, target):
        output_img, ouput_theta = output
        smoothness_loss = smoothness(ouput_theta)
        img_loss = loss(output_img, target)
        return img_loss# + smoothness_loss * 0.1

    for epoch in range(1, epochs + 1):
        utils.train(model, calculate_loss, optimizer, epoch, train_loader, valid_loader, writer=writer, checkpoint_name='stn_best.pth', scheduler=scheduler)
    
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
        '--subset', type=str, choices=data.lesion_subsets, default='isic', help='which dataset to use'
    )
    parser.add_argument(
        '--log-name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), help='name of folder where checkpoints are stored',
    )
    args = parser.parse_args()
    utils.save_args(args, 'stn')
    train_stn(**vars(args))