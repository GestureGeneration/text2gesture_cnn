#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
from utils.speaker_const import SPEAKERS_CONFIG


class wp_dataset(torch.utils.data.Dataset):
    def __init__(self, path, fnames, speaker):
        self.path = path
        self.fnames = fnames
        self.speaker = speaker
 
    def __len__(self):
        return len(self.fnames)
 
    def __getitem__(self, idx):
        npz = np.load(self.path / self.fnames[idx])
        wvec = npz['wvec']    # shape = (frames, 300)
        poses = npz['poses']  # shape = (frames, 2, 49)
        poses = np.reshape(poses, (poses.shape[0], poses.shape[1] * poses.shape[2]))  # shape = (frames, 98)
        # Standardization using SPEAKERS_CONFIG
        poses = (poses - SPEAKERS_CONFIG[self.speaker]['mean']) / (SPEAKERS_CONFIG[self.speaker]['std'] + np.finfo(float).eps)
        wvec = np.transpose(wvec, (1, 0))    # shape = (300, frames)
        poses = np.transpose(poses, (1, 0))  # shape = (98, frames)
        return torch.Tensor(wvec), torch.Tensor(poses)


def get_datalist(df_path, min_ratio=0.75, max_ratio=1.25):
    df = pd.read_csv(df_path)
    speaker = df['speaker'][0]

    shoulder_w = np.sqrt((SPEAKERS_CONFIG[speaker]['median'][4] - SPEAKERS_CONFIG[speaker]['median'][1]) ** 2
                         + (SPEAKERS_CONFIG[speaker]['median'][53] - SPEAKERS_CONFIG[speaker]['median'][50]) ** 2)
    min_w = shoulder_w * min_ratio
    max_w = shoulder_w * max_ratio
    shoulder_cond = (min_w < df['min_sh_width']) & (df['max_sh_width'] < max_w)

    file_exist = df['npz_fn'].notnull()
    train_list = df[(df['dataset'] == 'train') & shoulder_cond & file_exist]['npz_fn']
    dev_list = df[(df['dataset'] == 'dev') & shoulder_cond & file_exist]['npz_fn']

    print('train: ', len(train_list), ' / ', len(df[df['dataset'] == 'train']))
    print('dev: ', len(dev_list), ' / ', len(df[df['dataset'] == 'dev']))
    return train_list.to_list(), dev_list.to_list()
