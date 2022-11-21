
import torch
import numpy as np
import pandas as pd
import librosa
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from utils.speaker_const import SPEAKERS_CONFIG
from transformer.model import TransformerEnc
from transformer.mel_features import log_mel_spectrogram


def raw_repr(path, sr=None):
    wav, sr = librosa.load(path, sr=sr, mono=True)
    return wav, sr


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


def prediction_trans_enc(base_path, modality, test_speaker, test_dir, model_dir, model_path, outdir_path, gpu_num=0):
    # Save setting
    out_path = Path(outdir_path) / (test_speaker + '_by_' + model_dir + '_' + test_dir)

    os.makedirs(out_path, exist_ok=True)
    with open(Path(out_path).joinpath('setting.txt'), mode='w') as f:
        f.write('Base path: {}\n'.format(base_path))
        f.write('Modality: {}\n'.format(modality))
        f.write('Test speaker: {}\n'.format(test_speaker))
        f.write('Test dir: {}\n'.format(test_dir))
        f.write('Model dir: {}\n'.format(model_dir))
        f.write('Model path: {}\n'.format(model_path))

    # modality
    if modality == 'text':
        input_dim = 300
    elif modality == 'audio':
        input_dim = 64
    else:
        print('Invalid modality!')
        return

    # Get data list
    df_path = Path(base_path) / test_speaker / test_dir / 'test.csv'
    test_list = get_test_datalist(df_path)

    # Load model speaker
    exp_df = pd.read_csv(Path(model_path) / model_dir / 'experimental_settings.csv', index_col=0, header=None)
    model_speaker = exp_df.loc['speaker', 1]

    # Load trained generator model
    weights = torch.load(Path(model_path) / model_dir / 'trained-generator.pth',
                         map_location=lambda storage, loc: storage)
    model = TransformerEnc(
        # Default parameters in Transformer
        dim_input=input_dim, dim_model=512, num_heads=8, num_encoder_layers=4, dropout_p=0.1, dim_out=98,
    )

    # Set the GPU usage
    device = torch.device('cuda:' + str(gpu_num))
    print('Device: ', device)

    # Set trained weight
    model.load_state_dict(weights)
    model.to(device)
    model.eval()  # evaluation mode

    # Prediction
    npz_path = Path(base_path) / test_speaker / test_dir / 'text-pose-npz'
    audio_path = Path(base_path) / test_speaker / test_dir / 'audio'
    for fn in tqdm(test_list):
        # Load data
        npz = np.load(npz_path / fn)
        if modality == 'text':
            x = npz['wvec']  # shape = (frames, 300)
            x = np.transpose(x, (1, 0))  # shape = (300, frames)
        elif modality == 'audio':
            audio_fn = fn[:-3] + 'wav'
            wav, _ = raw_repr(audio_path / audio_fn, 16000)  # shape=(16000 * sec, )
            x = log_mel_spectrogram(wav, audio_sample_rate=16000, log_offset=1e-6,
                                    window_length_secs=0.025, hop_length_secs=1/15,
                                    num_mel_bins=64, lower_edge_hertz=125.0, upper_edge_hertz=7500.0)
            x = np.transpose(x, (1, 0))  # shape = (64, 100 * sec)
        inputs = torch.Tensor([x]).to(device)  # shape = (1, dim_input, seq_len)

        # Model prediction
        with torch.no_grad():
            pred = model(inputs.permute(0, 2, 1))  # pred: shape = (seq_len, 1, num_out)
            gesture = pred.cpu().numpy()[:, 0, :]  # shape = (seq_len, num_out=98)

        # De-normalizing gestures using SPEAKERS_CONFIG
        gesture = (SPEAKERS_CONFIG[model_speaker]['std'] + np.finfo(float).eps) * gesture + SPEAKERS_CONFIG[model_speaker]['mean']
        gesture = np.reshape(gesture, (-1, 2, 49))  # shape = (frames, 2, 49)

        # Saving
        np.save(out_path / fn[:-4], gesture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text to Gesture Generation by PyTorch')
    parser.add_argument('--base_path', type=str, default='~/Gestures/', help='gesture base path')
    parser.add_argument('--modality', type=str, default='text', help='text or audio')
    parser.add_argument('--test_speaker', type=str, default='oliver', help='speaker name for test')
    parser.add_argument('--test_dir', type=str, default='test-192', help='test file directory name')
    parser.add_argument('--model_dir', type=str, default='oliver_YYYYMMDD-AAAAAA',
                        help='directory name of trained model')
    parser.add_argument('--model_path', type=str, default='./out/',
                        help='directory path to training result')
    parser.add_argument('--outdir_path', type=str, default='~/test_gesture_out/text2gesture/',
                        help='directory path of outputs')
    args = parser.parse_args()

    prediction_trans_enc(args.base_path, args.modality, args.test_speaker, args.test_dir, args.model_dir,
                           args.model_path, args.outdir_path)
