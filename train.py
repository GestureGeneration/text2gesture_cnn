#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim

import time
import os
import csv
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.torch_dataset import get_datalist, wp_dataset
from utils.model import UnetDecoder, PatchGan


def get_argument():
    """
    Experimental setting

    Returns
    -------
    args: Namespace
        Experimental parameters from command line
    """
    parser = argparse.ArgumentParser(description='Text to Gesture Generation by PyTorch')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--speaker', type=str, default='oliver', help='choose speaker name')
    parser.add_argument('--no_screening', action='store_true', help='Not use data screening')
    parser.add_argument('--gan_loss', action='store_true', help='Use GAN loss')
    parser.add_argument('--lam_p', type=float, default=1., help='coefficient of pose loss')
    parser.add_argument('--lam_m', type=float, default=1., help='coefficient of motion loss')
    parser.add_argument('--lam_g', type=float, default=1., help='coefficient of GAN loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for training')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--gpu_num', type=int, default='0', help='GPU number')
    parser.add_argument('--base_path', type=str, default='~/Gestures/', help='gesture base path')
    parser.add_argument('--train_dir', type=str, default='train-64', help='training file directory')
    parser.add_argument('--outdir_path', type=str, default='./out/', help='directory path of outputs')
    parser.add_argument('--model_save_interval', type=int, default='10', help='Interval for saving model')
    args = parser.parse_args()
    return args


def write_parameters(args, outdir_path):
    """
    Write hyperparameter settings to csv file

    Parameters
    ----------
    args: Namespace
        Experimental Settings
    outdir_path: string
        Output path
    """
    fout = open(Path(outdir_path).joinpath('experimental_settings.csv'), "wt")
    csvout = csv.writer(fout)
    print('*' * 50)
    print('Parameters')
    print('*' * 50)
    for arg in dir(args):
        if not arg.startswith('_'):
            csvout.writerow([arg,  str(getattr(args, arg))])
            print('%-25s %-25s' % (arg, str(getattr(args, arg))))


def train(args, outdir_path):
    """
    Main function for training

    Returns
    -------
    net: type(model)
        Trained model at final iteration
    """

    # Load the dataset
    df_path = Path(args.base_path) / args.speaker / args.train_dir / 'train.csv'
    dataset_path = Path(args.base_path) / args.speaker / args.train_dir / 'text-pose-npz'
    if args.no_screening:
        train_list, dev_list = get_datalist(df_path, min_ratio=-np.inf, max_ratio=np.inf)
    else:
        train_list, dev_list = get_datalist(df_path)
    train_num, val_num = len(train_list), len(dev_list)
    print('Dataset size: {} (train), {} (validation)'.format(train_num, val_num))

    train_dataset = wp_dataset(dataset_path, train_list, args.speaker)
    val_dataset = wp_dataset(dataset_path, dev_list, args.speaker)

    # DataLoaders
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    print('Complete the preparing dataset...')

    # Set the GPU usage
    device = torch.device('cuda:' + str(args.gpu_num) if args.device == 'cuda' else 'cpu')
    print('Device: ', device)

    # Set the network
    # Gesture Generator
    g_net = UnetDecoder(300, 300)
    g_net.to(device)
    g_optim = optim.Adam(g_net.parameters(), lr=args.lr)

    # Define Loss function
    l1_criterion = nn.L1Loss()

    # Training loop
    start_time = time.time()
    ite = 0
    history = []
    g_net.train(True)

    for epoch in range(args.epochs):

        running_sample = train_frame_num = 0
        train_pose_sum, train_pose_sq_sum = np.zeros(98), np.zeros(98)
        train_g_pose_l1 = train_g_motion_l1 = 0.0

        # ********** Training Phase **********
        # Sample minibatch from DataLoader
        for (x_train, t_train) in tqdm(train_loader):
            ite += 1
            inputs, corrects = x_train.to(device), t_train.to(device)

            # Generator training
            g_optim.zero_grad()
            g_out = g_net(inputs)  # shape = (batch, 98, frames=64)
            g_pose_l1 = l1_criterion(g_out, corrects)  # L1 Loss of each coordinate
            # L1 loss of temporal difference
            g_motion_l1 = l1_criterion(g_out[:, :, 1:] - g_out[:, :, :-1], corrects[:, :, 1:] - corrects[:, :, :-1])
            g_loss = args.lam_p * g_pose_l1 + args.lam_m * g_motion_l1
            g_loss.backward()
            g_optim.step()

            # Record running loss and prediction
            poses = np.reshape(g_out.detach().cpu().numpy().transpose(0, 2, 1), (-1, 98))
            train_pose_sum += poses.sum(axis=0)
            train_pose_sq_sum += (poses ** 2).sum(axis=0)
            train_frame_num += len(poses)
            train_g_pose_l1 += g_pose_l1.item() * len(inputs)
            train_g_motion_l1 += g_motion_l1.item() * len(inputs)
            running_sample += len(inputs)

        # ********** Logging and Validation Phase **********
        g_net.train(False)
        val_frame_num = 0
        val_pose_sum, val_pose_sq_sum = np.zeros(98), np.zeros(98)
        val_g_pose_l1 = val_g_motion_l1 = 0.0

        for x_val, t_val in val_loader:
            inputs, corrects = x_val.to(device), t_val.to(device)

            with torch.no_grad():
                # Generator Calculation
                g_out = g_net(inputs)
                g_pose_l1 = l1_criterion(g_out, corrects)
                g_motion_l1 = l1_criterion(g_out[:, :, 1:] - g_out[:, :, :-1],
                                           corrects[:, :, 1:] - corrects[:, :, :-1])

                # Record running loss and prediction
                poses = np.reshape(g_out.detach().cpu().numpy().transpose(0, 2, 1), (-1, 98))
                val_pose_sum += poses.sum(axis=0)
                val_pose_sq_sum += (poses ** 2).sum(axis=0)
                val_frame_num += len(poses)
                val_g_pose_l1 += g_pose_l1.item() * len(inputs)
                val_g_motion_l1 += g_motion_l1.item() * len(inputs)

        g_net.train(True)

        # Record training log
        train_pose_std = np.mean(np.sqrt(train_pose_sq_sum / train_frame_num
                                         - (train_pose_sum / train_frame_num) ** 2))
        val_pose_std = np.mean(np.sqrt(val_pose_sq_sum / val_frame_num - (val_pose_sum / val_frame_num) ** 2))
        record = {'epoch': epoch + 1, 'iteration': ite,
                  'train_pose_std': train_pose_std, 'val_pose_std': val_pose_std,
                  'train_g_pose_l1': train_g_pose_l1 / running_sample,
                  'train_g_motion_l1': train_g_motion_l1 / running_sample,
                  'train_g_loss': (args.lam_p * train_g_pose_l1 + args.lam_m * train_g_motion_l1) / running_sample,
                  'val_g_pose_l1': val_g_pose_l1 / val_num,
                  'val_g_motion_l1': val_g_motion_l1 / val_num,
                  'val_g_loss': (args.lam_p * val_g_pose_l1 + args.lam_m * val_g_motion_l1) / val_num}
        history.append(record)
        print(record, flush=True)

        # Save models
        if (epoch + 1) % args.model_save_interval == 0:
            torch.save(g_net.state_dict(), Path(outdir_path).joinpath('generator-{}.pth'.format(epoch + 1)))
            pd.DataFrame.from_dict(history).to_csv(Path(outdir_path).joinpath('history.csv'))

    pd.DataFrame.from_dict(history).to_csv(Path(outdir_path).joinpath('history.csv'))

    # Training Time
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    # Save training time and dataset size
    with open(Path(outdir_path).joinpath('train_summary.txt'), mode='w') as f:
        f.write('Training size: {}, Val size: {}\n'.format(train_num, val_num))
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(elapsed_time // 60, elapsed_time % 60))

    return g_net


def train_gan(args, outdir_path):
    """
    Main function for training with GAN loss

    Returns
    -------
    net: type(model)
        Trained model at final iteration
    """

    # Load the dataset
    df_path = Path(args.base_path) / args.speaker / args.train_dir / 'train.csv'
    dataset_path = Path(args.base_path) / args.speaker / args.train_dir / 'text-pose-npz'
    if args.no_screening:
        train_list, dev_list = get_datalist(df_path, min_ratio=0.7, max_ratio=1.3)
    else:
        train_list, dev_list = get_datalist(df_path)
    train_num, val_num = len(train_list), len(dev_list)
    print('Dataset size: {} (train), {} (validation)'.format(train_num, val_num))
    train_dataset = wp_dataset(dataset_path, train_list, args.speaker)
    val_dataset = wp_dataset(dataset_path, dev_list, args.speaker)

    # DataLoaders
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                         drop_last=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    print('Complete the preparing dataset...')

    # Set the GPU usage
    device = torch.device('cuda:' + str(args.gpu_num) if args.device == 'cuda' else 'cpu')
    print('Device: ', device)

    # Set the network
    # Gesture Generator
    g_net = UnetDecoder(300, 300)
    g_net.to(device)
    g_optim = optim.Adam(g_net.parameters(), lr=args.lr)

    # Discriminator
    d_net = PatchGan(ndf=64)
    d_net.to(device)
    d_optim = optim.Adam(d_net.parameters(), lr=args.lr)

    # Define Loss function
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    # Training loop
    start_time = time.time()
    ite = 0
    history = []
    g_net.train(True)
    d_net.train(True)

    for epoch in range(args.epochs):

        running_sample = train_frame_num = 0
        train_pose_sum, train_pose_sq_sum = np.zeros(98), np.zeros(98)
        train_g_pose_l1 = train_g_motion_l1 = train_g_gan = train_d_real = train_d_fake = 0.0

        # ********** Training Phase **********
        # Sample minibatch from DataLoader
        for (x_train, t_train) in tqdm(train_loader):
            ite += 1
            inputs, corrects = x_train.to(device), t_train.to(device)

            # Generator training
            g_optim.zero_grad()
            g_out = g_net(inputs)  # shape = (batch, 98, frames=64)
            d_fake_out = d_net(g_out[:, :, 1:] - g_out[:, :, :-1])
            g_gan_loss = mse_criterion(torch.ones(d_fake_out.shape).to(device), d_fake_out)
            g_pose_l1 = l1_criterion(g_out, corrects)  # L1 Loss of each coordinate
            # L1 loss of temporal difference
            g_motion_l1 = l1_criterion(g_out[:, :, 1:] - g_out[:, :, :-1], corrects[:, :, 1:] - corrects[:, :, :-1])
            g_loss = args.lam_p * g_pose_l1 + args.lam_m * g_motion_l1 + args.lam_g * g_gan_loss
            g_loss.backward()
            g_optim.step()

            # Discriminator training
            d_optim.zero_grad()
            fake_d_input = g_out[:, :, 1:] - g_out[:, :, :-1]
            real_d_input = corrects[:, :, 1:] - corrects[:, :, :-1]
            d_real_out, d_fake_out = d_net(real_d_input), d_net(fake_d_input.detach())
            d_real_loss = mse_criterion(torch.ones(d_real_out.shape).to(device), d_real_out)
            d_fake_loss = mse_criterion(torch.zeros(d_fake_out.shape).to(device), d_fake_out)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Record running loss and prediction
            poses = np.reshape(g_out.detach().cpu().numpy().transpose(0, 2, 1), (-1, 98))
            train_pose_sum += poses.sum(axis=0)
            train_pose_sq_sum += (poses**2).sum(axis=0)
            train_frame_num += len(poses)
            train_g_pose_l1 += g_pose_l1.item() * len(inputs)
            train_g_motion_l1 += g_motion_l1.item() * len(inputs)
            train_g_gan += g_gan_loss.item() * len(inputs)
            train_d_real += d_real_loss.item() * len(inputs)
            train_d_fake += d_fake_loss.item() * len(inputs)
            running_sample += len(inputs)

        # ********** Logging and Validation Phase **********
        g_net.train(False)
        d_net.train(False)
        val_frame_num = 0
        val_pose_sum, val_pose_sq_sum = np.zeros(98), np.zeros(98)
        val_g_pose_l1 = val_g_motion_l1 = val_g_gan = val_d_real = val_d_fake = 0.0

        for x_val, t_val in val_loader:
            inputs, corrects = x_val.to(device), t_val.to(device)

            with torch.no_grad():
                # Generator Calculation
                g_out = g_net(inputs)
                g_pose_l1 = l1_criterion(g_out, corrects)
                g_motion_l1 = l1_criterion(g_out[:, :, 1:] - g_out[:, :, :-1],
                                           corrects[:, :, 1:] - corrects[:, :, :-1])
                d_fake_out = d_net(g_out[:, :, 1:] - g_out[:, :, :-1])
                g_gan_loss = mse_criterion(torch.ones(d_fake_out.shape).to(device), d_fake_out)

                # Discriminator Calculation
                fake_d_input = g_out[:, :, 1:] - g_out[:, :, :-1]
                real_d_input = corrects[:, :, 1:] - corrects[:, :, :-1]
                d_real_out, d_fake_out = d_net(real_d_input), d_net(fake_d_input)
                d_real_loss = mse_criterion(torch.ones(d_real_out.shape).to(device), d_real_out)
                d_fake_loss = mse_criterion(torch.zeros(d_fake_out.shape).to(device), d_fake_out)

                # Record running loss and prediction
                poses = np.reshape(g_out.detach().cpu().numpy().transpose(0, 2, 1), (-1, 98))
                val_pose_sum += poses.sum(axis=0)
                val_pose_sq_sum += (poses ** 2).sum(axis=0)
                val_frame_num += len(poses)
                val_g_pose_l1 += g_pose_l1.item() * len(inputs)
                val_g_motion_l1 += g_motion_l1.item() * len(inputs)
                val_g_gan += g_gan_loss.item() * len(inputs)
                val_d_real += d_real_loss.item() * len(inputs)
                val_d_fake += d_fake_loss.item() * len(inputs)

        g_net.train(True)
        d_net.train(True)

        # Record training log
        train_pose_std = np.mean(np.sqrt(train_pose_sq_sum / train_frame_num
                                         - (train_pose_sum / train_frame_num)**2))
        val_pose_std = np.mean(np.sqrt(val_pose_sq_sum / val_frame_num - (val_pose_sum / val_frame_num)**2))
        record = {'epoch': epoch + 1, 'iteration': ite,
                  'train_pose_std': train_pose_std, 'val_pose_std': val_pose_std,
                  'train_g_pose_l1': train_g_pose_l1 / running_sample,
                  'train_g_motion_l1': train_g_motion_l1 / running_sample,
                  'train_g_gan': train_g_gan / running_sample,
                  'train_g_loss': (args.lam_p * train_g_pose_l1 + args.lam_m * train_g_motion_l1 + args.lam_g * train_g_gan) / running_sample,
                  'train_d_real': train_d_real / running_sample, 'train_d_fake': train_d_fake / running_sample,
                  'train_d_loss': (train_d_real + train_d_fake) / running_sample,
                  'val_g_pose_l1': val_g_pose_l1 / val_num,
                  'val_g_motion_l1': val_g_motion_l1 / val_num,
                  'val_g_gan': val_g_gan / val_num,
                  'val_g_loss': (args.lam_p * val_g_pose_l1 + args.lam_m * val_g_motion_l1 + args.lam_g * val_g_gan) / val_num,
                  'val_d_real': val_d_real / val_num, 'val_d_fake': val_d_fake / val_num,
                  'val_d_loss': (val_d_real + val_d_fake) / val_num}
        history.append(record)
        print(record, flush=True)

        # Save models
        if (epoch + 1) % args.model_save_interval == 0:
            torch.save(g_net.state_dict(), Path(outdir_path).joinpath('generator-{}.pth'.format(epoch + 1)))
            torch.save(d_net.state_dict(), Path(outdir_path).joinpath('discriminator-{}.pth'.format(epoch + 1)))
            pd.DataFrame.from_dict(history).to_csv(Path(outdir_path).joinpath('history.csv'))

    pd.DataFrame.from_dict(history).to_csv(Path(outdir_path).joinpath('history.csv'))

    # Training Time
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    # Save training time and dataset size
    with open(Path(outdir_path).joinpath('train_summary.txt'), mode='w') as f:
        f.write('Training size: {}, Val size: {}\n'.format(train_num, val_num))
        f.write('Training complete in {:.0f}m {:.0f}s\n'.format(elapsed_time // 60, elapsed_time % 60))

    return [g_net, d_net]


if __name__ == '__main__':
    time_stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    args = get_argument()

    # Make directory to save results
    outdir_path = Path(args.outdir_path) / (args.speaker + '_' + time_stamp)
    os.makedirs(outdir_path, exist_ok=True)
    write_parameters(args, outdir_path)

    # Check GPU / CPU
    if not torch.cuda.is_available():
        args.device = 'cpu'

    # Swith the training function according to GAN loss usage
    if args.gan_loss:
        nets = train_gan(args, outdir_path)
        # Save trained network
        for name, net in zip(['generator', 'discriminator'], nets):
            torch.save(net.state_dict(), Path(outdir_path).joinpath('trained-{}.pth'.format(name)))
    else:
        net = train(args, outdir_path)
        # Save trained network
        torch.save(net.state_dict(), Path(outdir_path).joinpath('trained-{}.pth'.format('generator')))
