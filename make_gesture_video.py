import os
import subprocess
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from test import get_test_datalist
from utils.gesture_plot import save_video
from utils.speaker_const import SPEAKERS_CONFIG


def make_gesture_video(trg_speaker, i, face, pose_gt, pose_fake, audio_fn, video_out_path, tmp_path):
    """
    # Decide position randomly
    if np.random.randint(2) == 0:
        video_name = trg_speaker + '_{:03d}_right.mp4'.format(i)
        poses = np.array([pose_gt, pose_fake])  # Pose keypoints
        info = [video_name, 'real', 'fake']
    else:
        video_name = trg_speaker + '_{:03d}_left.mp4'.format(i)
        poses = np.array([pose_fake, pose_gt])  # Pose keypoints
        info = [video_name, 'fake', 'real']
    """

    video_name = trg_speaker + '_{:03d}_right.mp4'.format(i)
    poses = np.array([pose_gt, pose_fake])  # Pose keypoints
    info = [video_name, 'real', 'fake']

    # Save video
    save_video(poses, face, audio_fn, str(video_out_path / video_name), str(tmp_path), delete_tmp=False)

    return info


def get_test_df(df_path, min_ratio=0.75, max_ratio=1.25):
    df = pd.read_csv(df_path)
    speaker = df['speaker'][0]

    shoulder_w = np.sqrt((SPEAKERS_CONFIG[speaker]['median'][4] - SPEAKERS_CONFIG[speaker]['median'][1]) ** 2
                         + (SPEAKERS_CONFIG[speaker]['median'][53] - SPEAKERS_CONFIG[speaker]['median'][50]) ** 2)
    min_w = shoulder_w * min_ratio
    max_w = shoulder_w * max_ratio
    shoulder_cond = (min_w < df['min_sh_width']) & (df['max_sh_width'] < max_w)

    file_exist = df['npz_fn'].notnull()
    test_df = df[(df['dataset'] == 'test') & shoulder_cond & file_exist]
    return test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making gesture videos')
    parser.add_argument('--base_path', type=str, default='~/Gestures/', help='gesture dataset base path')
    parser.add_argument('--test_out_path', type=str, default='./out_test/', help='directory path of test output')
    parser.add_argument('--test_out_dir', type=str, default='oliver_by_oliver_YYYYMMDD-AAAAAA_test-192',
                        help='directory name of output gestures')
    parser.add_argument('--video_out_path', type=str, default='./out_video/', help='directory path of output videos')

    parser.add_argument('--video_num', type=int, default=50, help='number of videos')
    parser.add_argument('--tmp_path', type=str, default='./tmp', help='temporary directory path')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    test_out_path = Path(args.test_out_path)
    test_out_dir = args.test_out_dir
    video_out_path = Path(args.video_out_path) / 'text2gesture' / test_out_dir
    os.makedirs(video_out_path, exist_ok=True)

    NUM = 50
    tmp_path = Path(args.tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    # Extract information
    trg_speaker = test_out_dir.split('_')[0]
    model_speaker = test_out_dir.split('_')[2]
    frame_num = int(test_out_dir.split('-')[-1])
    test_dir_name = test_out_dir.split('_')[-1]

    # Video of Text2Gesture and GroundTruth
    cols = ['test_fn', 'video_name', 'left', 'right'] + ['word-' + str(i + 1) for i in range(frame_num)]
    data_path = base_path / trg_speaker / test_dir_name
    audio_path = Path(data_path) / 'audio'
    df_path = data_path / 'test.csv'

    # Test file list
    test_list = get_test_datalist(df_path, min_ratio=0.75, max_ratio=1.25)

    # Test gesture file path
    t2g_test_pose_path = test_out_path / test_out_dir

    df_record = pd.DataFrame(index=[], columns=cols)

    for i, npz_fn in tqdm(enumerate(test_list.values[:NUM])):
        # Ground Truth
        npz_gt = np.load(Path(data_path) / 'text-pose-npz' / npz_fn)
        pose_gt = npz_gt['poses']
        word_gt = list(npz_gt['words'])

        # Face keypoints
        face = np.array([npz_gt['face'], npz_gt['face']])

        # Audio file names
        audio_name = npz_fn[:-3] + 'wav'

        # Text2Gesture Prediction
        pose_t2g = np.load(t2g_test_pose_path / (npz_fn[:-1] + 'y'))
        pose_t2g = SPEAKERS_CONFIG[model_speaker]['scale_factor'] / SPEAKERS_CONFIG[trg_speaker][
            'scale_factor'] * pose_t2g

        info = make_gesture_video(trg_speaker, i, face, pose_gt, pose_t2g, str(audio_path / audio_name),
                                  video_out_path, tmp_path)

        # Save record
        record = pd.Series([npz_fn] + info + word_gt, index=cols)
        df_record = df_record.append(record, ignore_index=True)

    df_record.to_csv(video_out_path / 't2g.csv', index=False)

    # --------------------------------------------------
    # Save original video
    cols = ['test_fn', 'video_name'] + ['word-' + str(i + 1) for i in range(frame_num)]

    # Test file list
    test_df = get_test_df(df_path)

    # Save directory
    video_out_path = Path(args.video_out_path) / 'original' / trg_speaker
    os.makedirs(video_out_path, exist_ok=True)

    df_record = pd.DataFrame(index=[], columns=cols)

    for i in tqdm(range(NUM)):
        npz_fn = test_df.iloc[i]['npz_fn']
        npz_gt = np.load(Path(data_path) / 'text-pose-npz' / npz_fn)
        word = list(npz_gt['words'])
        # print(npz_fn)

        zero = datetime.datetime.strptime('0', '%S')
        start = datetime.datetime.strptime(test_df.iloc[i]['start'], '%H:%M:%S.%f') - zero
        end = datetime.datetime.strptime(test_df.iloc[i]['end'], '%H:%M:%S.%f') - zero
        video_fn = base_path / trg_speaker / 'videos' / test_df.iloc[i]['video_fn']
        # print(video_fn, start, end)

        video_out_fn = video_out_path / (trg_speaker + '_{:03d}.mp4'.format(i))
        # print(video_out_fn)

        # Extract video
        cmd = 'ffmpeg -i {} -ss {} -to {} -y {}'.format(str(video_fn), start, end, str(video_out_fn))
        # print(cmd)
        subprocess.call(cmd, shell=True)

        # Save record
        record = pd.Series([npz_fn, video_out_fn] + word, index=cols)
        df_record = df_record.append(record, ignore_index=True)

    df_record.to_csv(video_out_path / 'original.csv', index=False)
