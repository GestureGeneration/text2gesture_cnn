
import torch
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from utils.speaker_const import SPEAKERS_CONFIG
from utils.model import UnetDecoder


def get_test_datalist(df_path, min_ratio=-np.inf, max_ratio=np.inf):
    df = pd.read_csv(df_path)
    speaker = df['speaker'][0]

    shoulder_w = np.sqrt((SPEAKERS_CONFIG[speaker]['median'][4] - SPEAKERS_CONFIG[speaker]['median'][1]) ** 2
                         + (SPEAKERS_CONFIG[speaker]['median'][53] - SPEAKERS_CONFIG[speaker]['median'][50]) ** 2)
    min_w = shoulder_w * min_ratio
    max_w = shoulder_w * max_ratio
    shoulder_cond = (min_w < df['min_sh_width']) & (df['max_sh_width'] < max_w)

    file_exist = df['npz_fn'].notnull()
    test_list = df[(df['dataset'] == 'test') & shoulder_cond & file_exist]['npz_fn']
    return test_list


def prediction(base_path, test_speaker, test_dir, model_dir, model_path, outdir_path):
    # Save setting
    out_path = Path(outdir_path) / (test_speaker + '_by_' + model_dir + '_' + test_dir)

    os.makedirs(out_path, exist_ok=True)
    with open(Path(out_path).joinpath('setting.txt'), mode='w') as f:
        f.write('Base path: {}\n'.format(base_path))
        f.write('Test speaker: {}\n'.format(test_speaker))
        f.write('Test dir: {}\n'.format(test_dir))
        f.write('Model dir: {}\n'.format(model_dir))
        f.write('Model path: {}\n'.format(model_path))

    # Get data list
    df_path = Path(base_path) / test_speaker / test_dir / 'test.csv'
    test_list = get_test_datalist(df_path)

    # Load model speaker
    exp_df = pd.read_csv(Path(model_path) / model_dir / 'experimental_settings.csv', index_col=0, header=None)
    model_speaker = exp_df.loc['speaker', 1]

    # Load trained generator model
    weights = torch.load(Path(model_path) / model_dir / 'trained-generator.pth',
                         map_location=lambda storage, loc: storage)
    model = UnetDecoder(300, 300)
    # Set trained weight
    model.load_state_dict(weights)

    # Prediction
    dataset_path = Path(base_path) / test_speaker / test_dir / 'text-pose-npz'
    for fn in tqdm(test_list):
        # Load word vectors
        npz = np.load(dataset_path / fn)
        wvec = npz['wvec']  # shape = (frames, 300)
        wvec = np.transpose(wvec, (1, 0))  # shape = (300, frames)
        inputs = torch.Tensor([wvec])

        # Model prediction
        with torch.no_grad():
            outputs = model(inputs)
        gesture = np.transpose(outputs.numpy()[0], (1, 0))  # shape = (frames, 98)

        # De-normalizing gestures using SPEAKERS_CONFIG
        gesture = (SPEAKERS_CONFIG[model_speaker]['std'] + np.finfo(float).eps) * gesture + SPEAKERS_CONFIG[model_speaker]['mean']
        gesture = np.reshape(gesture, (-1, 2, 49))  # shape = (frames, 2, 49)

        # Saving
        np.save(out_path / fn[:-4], gesture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text to Gesture Generation by PyTorch')
    parser.add_argument('--base_path', type=str, default='~/Gestures/', help='gesture base path')
    parser.add_argument('--test_speaker', type=str, default='oliver', help='speaker name for test')
    parser.add_argument('--test_dir', type=str, default='test-192', help='test file directory name')
    parser.add_argument('--model_dir', type=str, default='oliver_YYYYMMDD-AAAAAA',
                        help='directory name of trained model')
    parser.add_argument('--model_path', type=str, default='./out/',
                        help='directory path to training result')
    parser.add_argument('--outdir_path', type=str, default='~/test_gesture_out/text2gesture/',
                        help='directory path of outputs')
    args = parser.parse_args()

    prediction(args.base_path, args.test_speaker, args.test_dir, args.model_dir, args.model_path, args.outdir_path)
