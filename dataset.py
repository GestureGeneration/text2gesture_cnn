import numpy as np
import pandas as pd
import fasttext
import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import subprocess
import cv2
from decimal import Decimal

WORD_CSV_DIR = 'words'


class FasttextVectorizer:
    def __init__(self, model_path, dim=300):
        # Load the trained model of fasttext
        self.model = fasttext.load_model(model_path)
        self.dim = dim

    def get_vec(self, corpus):
        # Return embedded feature vectors from word list
        vec = []
        if len(corpus) != 0:
            for p in corpus:
                vec.append(self.model.get_word_vector(p))
        else:
            vec.append(np.zeros(self.dim))
        return vec


class WordEmb:
    def __init__(self, model_path, df, speaker_path, dim=300):
        self.vectorizer = FasttextVectorizer(model_path, dim)

        self.speaker_path = speaker_path
        video_names = df['video_fn'].unique()
        self.word_df_dict = {}
        for vn in video_names:
            video_name = vn.split('.')[0]
            # Read utterance information for each frame
            csv_name = '15FPS-{}.csv'.format(video_name)
            fps_df = pd.read_csv(speaker_path / WORD_CSV_DIR / csv_name)
            self.word_df_dict[video_name] = fps_df

    def get_wordemb(self, df, i, frames):
        zero = datetime.datetime.strptime('0', '%S')
        start = datetime.datetime.strptime(df['start'][i], '%H:%M:%S.%f') - zero
        # Get start time of pronounce
        start_sec = start.total_seconds()
        # Calculate starting frame number
        start_frame = round(start_sec * 15)

        video_name = df['video_fn'][i].split('.')[0]

        # Get utterance words for each frame
        fps_df = self.word_df_dict[video_name]
        word_list = fps_df['word'][start_frame:start_frame+frames].copy(deep=True).to_list()
        # Pad the data if the number of frames is insufficient (when extracting the data near the end of video)
        if not len(word_list) == frames:
            print('not length = {} ({})'.format(frames, len(word_list)))
            word_list.extend(['<BLANK>'] * (frames - len(word_list)))
        # Transform '<BLANK>'
        words = ['' if w == '<BLANK>' else w for w in word_list]
        # Vectorize the words
        vec = self.vectorizer.get_vec(words)
        return word_list, vec


def save_voice(speaker_path, i, df, fname, frames, voice_path):
    zero = datetime.datetime.strptime('0', '%S')
    start = datetime.datetime.strptime(df['start'][i], '%H:%M:%S.%f') - zero

    # Get start time and end time of pronounce
    start_sec = start.total_seconds()
    # Get total time (sec) of pronounce
    total_sec = 1./15 * frames

    # Path
    video_path = speaker_path / 'videos' / df['video_fn'][i]
    voice_name = fname + '.wav'
    voice_path = voice_path / voice_name

    # Save 'wav' file with the same name with 'npz' by ffmpeg
    cmd = 'ffmpeg -i "{}" -ss {} -t {} -ab 160k -ac 2 -ar {} -vn "{}" -y -loglevel warning'.format(str(video_path), start_sec, total_sec, 44100, str(voice_path))
    subprocess.call(cmd, shell=True)


class KptExtractor:
    def __init__(self, df, speaker_path):
        video_names = df['video_fn'].unique()
        self.speaker_path = speaker_path
        self.kpt_face_dict = {}
        for vn in video_names:
            video_name = vn.split('.')[0]
            keypoints_all_path = self.speaker_path / 'keypoints_all' / video_name
            # Get file names
            proc = subprocess.run(['ls', str(keypoints_all_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            fstr = proc.stdout.decode('utf8')
            flist = fstr.split('\n')
            # Extract keypoint data for face
            face_kpts_name = [[f] for f in flist if 'face' in f]
            # Calculate time (sec) from extracted file name
            face_kpts_time = [[float(Decimal(float(t[0].split('_')[-3]) * 60.0 + float(t[0].split('_')[-2])).quantize(Decimal('0.000001')))] for t in face_kpts_name]
            if len(face_kpts_name) != 0:
                # Concatenate
                face_kpts_list = np.concatenate([face_kpts_name, face_kpts_time], 1)
                # Sorting based on the frame number
                face_kpts_list = face_kpts_list[np.argsort(face_kpts_list[:, 1].astype(float))]
                self.kpt_face_dict[video_name] = face_kpts_list

    def get_kpts(self, df, i, frames, face_calc=True):
        pose_arr = np.empty((0, 2, 49), int)
        face_arr = np.empty((0, 2, 70), int)

        video_name = df['video_fn'][i].split('.')[0]
        # Get start time
        start = df['start'][i].split(':')
        start_sec = float(start[-2]) * 60.0 + float(start[-1])  # Transform to [sec]
        # Get data after the starting time
        frame_tf = self.kpt_face_dict[video_name][:, 1].astype(float) > start_sec
        start_face_kpts = self.kpt_face_dict[video_name][frame_tf]

        # Path for keypoint data
        keypoints_simple_path = self.speaker_path / 'keypoints_simple' / video_name
        keypoints_all_path = self.speaker_path / 'keypoints_all' / video_name

        for s in start_face_kpts[:frames]:
            # Keypoint for body
            pose_kpt = s[0][:-9] + '.txt'
            np_pose = np.loadtxt(keypoints_simple_path / pose_kpt)
            # Deletes two eyes and nose
            np_pose = np.delete(np_pose, [7, 8, 9], 1)  # 7 is nose, 8,9 are eyes
            base_p = np_pose[:, 0]
            np_pose = np_pose - np.reshape(base_p, (2, 1))
            pose_arr = np.append(pose_arr, np.reshape(np_pose, [1, 2, 49]), axis=0)

            # Keypoint for face
            if face_calc:
                fs = cv2.FileStorage(str(keypoints_all_path / s[0]), cv2.FILE_STORAGE_READ)
                face = fs.getNode("face_0").mat()
                x, y, _ = np.split(face, [1, 2], axis=2)
                # Use first person's data
                if not x.shape[0] == 1:
                    x, y = x[0], y[0]
                # Make (x, y)-array and arrange coordinate (the origin becomes base keypoint)
                face_2d = np.vstack((x - base_p[0], y - base_p[1]))
                face_arr = np.append(face_arr, np.reshape(face_2d, [1, 2, 70]), axis=0)

        return pose_arr, face_arr


def save_dataset(df, speaker_path, embed_path, save_path, voice=True, dataset='train', save_option=True, frames=64):

    npz_path = save_path / 'text-pose-npz'
    os.makedirs(npz_path, exist_ok=True)
    df['npz_fn'] = ''
    df['min_sh_width'] = 0.
    df['max_sh_width'] = 0.
    voice_path = save_path / 'audio'
    if voice:
        os.makedirs(voice_path, exist_ok=True)
        df['audio_fn'] = ''
    df['words'] = ''
    df['words_per_frame'] = ''

    # Load trained Word2Vec model
    print('loading word2vec')
    word_emb = WordEmb(embed_path, df, speaker_path)
    print('completed loading word2vec')

    kpt_ext = KptExtractor(df, speaker_path)

    for i in tqdm(range(len(df))):
        try:
            # word data
            words, wvec = word_emb.get_wordemb(df, i, frames)  # Get text vector
            df.at[i, 'words_per_frame'] = ' '.join(words)  # save words
            df.at[i, 'words'] = ' '.join(sorted(set(words), key=words.index))

            # keypoint data
            poses, face = kpt_ext.get_kpts(df, i, frames, face_calc=save_option)

            # Exception
            if poses.shape != (frames, 2, 49):
                continue
            elif save_option and face.shape != (frames, 2, 70):
                continue

            # Save
            fname = '{}-{:06d}'.format(df['dataset'][i], i)
            if save_option:
                np.savez(npz_path / fname, words=words, poses=poses, wvec=wvec, face=face)
            else:
                np.savez(npz_path / fname, poses=poses, wvec=wvec)
            df.at[i, 'npz_fn'] = fname + '.npz'
            shoulder_width = np.sqrt((poses[:, 0, 4] - poses[:, 0, 1])**2 + (poses[:, 1, 4] - poses[:, 1, 1])**2)
            df.at[i, 'min_sh_width'] = np.min(shoulder_width)
            df.at[i, 'max_sh_width'] = np.max(shoulder_width)

            # Save audio data
            if voice and os.path.isfile(speaker_path / 'videos' / df['video_fn'][i]):
                save_voice(speaker_path, i, df, fname, frames, voice_path)
                df.at[i, 'audio_fn'] = fname + '.wav'

        except Exception as e:
            print(e)
            continue

    df.to_csv(save_path / '{}.csv'.format(dataset))
    return df


def video_samples_train(df, base_path, speaker, num_frames=64):
    df = df[df['speaker'] == speaker]
    df = df[(df['dataset'] == 'train') | (df['dataset'] == 'dev')]
    speaker_path = base_path / speaker

    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [], 'video_fn': [], 'speaker': []}
    intervals = df['interval_id'].unique()

    total_frames = 0
    total_train_frames = 0
    total_dev_frames = 0
    for interval in tqdm(intervals):
        try:
            df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
            video_fn = df_interval.iloc[0]['video_fn']
            speaker_name = df_interval.iloc[0]['speaker']
            if len(df_interval) < num_frames:
                print("interval: %s, num frames: %s. skipped" % (interval, len(df_interval)))
                continue

            # word file exist?
            word_csv_name = '15FPS-{}.csv'.format(video_fn.split('.')[0])
            if not os.path.isfile(speaker_path / WORD_CSV_DIR / word_csv_name):
                continue
            total_frames += len(df_interval)
            if df_interval.iloc[0]['dataset'] == 'train':
                total_train_frames += len(df_interval)
            elif df_interval.iloc[0]['dataset'] == 'dev':
                total_dev_frames += len(df_interval)

            for idx in range(0, len(df_interval) - num_frames, 5):
                sample = df_interval[idx:idx + num_frames]
                data_dict["dataset"].append(df_interval.iloc[0]['dataset'])
                data_dict["start"].append(sample.iloc[0]['pose_dt'])
                data_dict["end"].append(sample.iloc[-1]['pose_dt'])
                data_dict["interval_id"].append(interval)
                data_dict["video_fn"].append(video_fn)
                data_dict["speaker"].append(speaker_name)
        except Exception as e:
            print(e)
            continue
    return pd.DataFrame.from_dict(data_dict), [total_frames, total_train_frames, total_dev_frames]


def video_samples_test(df, base_path, speaker, num_samples, num_frames=64):
    df = df[df['speaker'] == speaker]
    df = df[df['dataset'] == 'test']
    speaker_path = base_path / speaker

    df['ones'] = 1
    grouped = df.groupby('interval_id').agg({'ones': sum}).reset_index()
    grouped = grouped[grouped['ones'] >= num_frames][['interval_id']]
    df = df.merge(grouped, on='interval_id')

    data_dict = {'dataset': [], 'start': [], 'end': [], 'interval_id': [], 'video_fn': [], 'speaker': []}
    intervals = df['interval_id'].unique()

    i = 0
    pbar = tqdm(total=num_samples)
    while i < num_samples:
        try:
            interval = intervals[np.random.randint(0, len(intervals))]
            df_interval = df[df['interval_id'] == interval].sort_values('frame_id', ascending=True)
            video_fn = df_interval.iloc[0]['video_fn']
            speaker_name = df_interval.iloc[0]['speaker']
            if len(df_interval) < num_frames:
                continue

            # video file exist?
            if not os.path.isfile(speaker_path / 'videos' / video_fn):
                continue

            idx = np.random.randint(0, len(df_interval) - num_frames + 1)
            sample = df_interval[idx:idx + num_frames]
            data_dict["dataset"].append(df_interval.iloc[0]['dataset'])
            data_dict["start"].append(sample.iloc[0]['pose_dt'])
            data_dict["end"].append(sample.iloc[-1]['pose_dt'])
            data_dict["interval_id"].append(interval)
            data_dict["video_fn"].append(video_fn)
            data_dict["speaker"].append(speaker_name)
            i += 1
            pbar.update(1)
        except Exception as e:
            print(e)
            continue
    pbar.close()
    return pd.DataFrame.from_dict(data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speaker', '--speaker', default='specific speaker name', required=True)
    parser.add_argument('-wordvec_file', '--wordvec_file', default='./wiki-news-300d-1M-subword.bin',
                        help='word vector file of firstText', required=True)
    parser.add_argument('-dataset_type', '--dataset_type', default='train', help='dataset type (train / test)',
                        required=True)
    parser.add_argument('-fs', '--frames', default=64, help='number of frames per sample', type=int)
    parser.add_argument('-s', '--samples', default=4096, help='number of samples for test data', type=int)
    args = parser.parse_args()

    speaker = args.speaker
    base_path = Path(args.base_path)
    save_path = base_path / speaker / (args.dataset_type + '-' + str(args.frames))
    speaker_path = base_path / speaker    # Path of speaker

    df = pd.read_csv(base_path / 'frames_df_10_19_19.csv')

    if args.dataset_type == 'train':
        df_samples, frames = video_samples_train(df, base_path, speaker, num_frames=args.frames)
    elif args.dataset_type == 'test':
        df_samples = video_samples_test(df, base_path, speaker, args.samples, num_frames=args.frames)
    del df

    if args.dataset_type == 'train':
        df_word = save_dataset(df_samples, speaker_path, args.wordvec_file, save_path, voice=False,
                               dataset=args.dataset_type, save_option=False, frames=args.frames)
    elif args.dataset_type == 'test':
        df_word = save_dataset(df_samples, speaker_path, args.wordvec_file, save_path, voice=True,
                               dataset=args.dataset_type, save_option=True, frames=args.frames)

    print('speaker: ', speaker)
    print('dataset_type: ', args.dataset_type)
    if args.dataset_type == 'train':
        print('Number of total train/dev frames: ', frames[0])
        print('Number of total train frames: ', frames[1])
        print('Number of total dev frames: ', frames[2])
        print('Number of train/dev samples: ', len(df_word))
        print('Number of train samples: ', len(df_word[(df_word['dataset'] == 'train')]))
        print('Number of dev samples: ', len(df_word[(df_word['dataset'] == 'dev')]))
    elif args.dataset_type == 'test':
        print('Number of test samples: ', len(df_word))
