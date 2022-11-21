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

from transformer.torch_dataset import get_datalist, wp_dataset, ap_dataset

from transformer.model import TransformerEnc


def get_argument():
    """
    Experimental setting

    Returns
    -------
    args: Namespace
        Experimental parameters from command line
    """
    parser = argparse.ArgumentParser(description='Gesture Generation by PyTorch')
    parser.add_argument('--modality', type=str, default='text', help='text or audio')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--speaker', type=str, default='oliver', help='choose speaker name')
    parser.add_argument('--no_screening', action='store_true', help='Not use data screening')
    parser.add_argument('--lam_p', type=float, default=1., help='coefficient of pose loss')
    parser.add_argument('--lam_m', type=float, default=1., help='coefficient of motion loss')
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


def train_trans_enc(args, outdir_path):
    """
    Main function for training

    Returns
    -------
    net: type(model)
        Trained model at final iteration
    """

    # Load the dataset
    df_path = Path(args.base_path) / args.speaker / args.train_dir / 'train.csv'
    dataset_path = Path(args.base_path) / args.speaker / args.train_dir / 'text-audio-pose-npz'
    if args.no_screening:
        train_list, dev_list = get_datalist(df_path, min_ratio=-np.inf, max_ratio=np.inf)
    else:
        train_list, dev_list = get_datalist(df_path)
    train_num, val_num = len(train_list), len(dev_list)
    print('Dataset size: {} (train), {} (validation)'.format(train_num, val_num))

    if args.modality == 'text':
        train_dataset = wp_dataset(dataset_path, train_list, args.speaker)
        val_dataset = wp_dataset(dataset_path, dev_list, args.speaker)
        input_dim = 300
    elif args.modality == 'audio':
        train_dataset = ap_dataset(dataset_path, train_list, args.speaker, hop_length_secs=1/15)
        val_dataset = ap_dataset(dataset_path, dev_list, args.speaker, hop_length_secs=1/15)
        input_dim = 64
    else:
        print('Invalid modality!')
        return

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

    g_net = TransformerEnc(
        # Default parameters in Transformer
        dim_input=input_dim, dim_model=512, num_heads=8, num_encoder_layers=4, dropout_p=0.1,
        dim_out=98,
    )
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
            # inputs: shape = (batch, 300/64, seq_len), corrects: shape = (batch, 98, seq_len)
            inputs, corrects = x_train.to(device), t_train.to(device)

            # Generator training
            g_optim.zero_grad()
            pred = g_net(inputs.permute(0, 2, 1))  # pred: shape = (seq_len, batch_size, num_out)
            # Permute pred to have batch size first again
            g_out = pred.permute(1, 2, 0)  # shape = (batch_size, num_out=98, seq_len)
            g_pose_l1 = l1_criterion(g_out, corrects)

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
                # Generator calculation
                pred = g_net(inputs.permute(0, 2, 1))  # pred: shape = (seq_len, batch_size, num_out)
                # Permute pred to have batch size first again
                g_out = pred.permute(1, 2, 0)  # shape = (batch_size, num_out=98, seq_len)
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

    # Model training
    net = train_trans_enc(args, outdir_path)

    # Save trained network
    torch.save(net.state_dict(), Path(outdir_path).joinpath('trained-{}.pth'.format('generator')))
