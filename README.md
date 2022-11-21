# Evaluation of Text-to-Gesture Generation Model Using Convolutional Neural Network

This repository contains the code for the text-to-gesture generation model using CNN.

The demonstration video of generated gestures is available at <https://youtu.be/JX4Gqy-Rmso>.

Eiichi Asakawa, Naoshi Kaneko, Dai Hasegawa, and Shinichi Shirakawa: **Evaluation of text-to-gesture generation model using convolutional neural network**, *Neural Networks*, Elsevier, Vol. 151, pp. 365-375, Jul. 2022. [[DOI](https://doi.org/10.1016/j.neunet.2022.03.041)]

If you use this code for your research, please cite our paper:

```
@article{AsakawaNEUNET2022,
    author = {Eiichi Asakawa and Naoshi Kaneko and Dai Hasegawa and Shinichi Shirakawa},
    title = {Evaluation of text-to-gesture generation model using convolutional neural network},
    journal = {Neural Networks},
    volume = {151},
    pages = {365--375},
    year = {2022},
    doi = {https://doi.org/10.1016/j.neunet.2022.03.041}
}
```

## Requirements
We used the [PyTorch](https://pytorch.org/) version 1.7.1 for neural network implementation. We tested the codes on the following environment:

- Ubuntu 16.04 LTS
- GPU: NVIDIA GeForce GTX 1080Ti
- Python environment: anaconda3-2020.07
    - [fasttext](https://fasttext.cc/)
    - cv2 (4.4.0)
- ffmpeg

## Preparation
1. Our code uses the speech and gesture dataset provided by Ginosar et al. Download the Speech2Gesture dataset by following the instruction "Download specific speaker data" in <https://github.com/amirbar/speech2gesture/blob/master/data/dataset.md>.

```
Shiry Ginosar, Amir Bar, Gefen Kohavi, Caroline Chan, Andrew Owens, and Jitendra Malik, "Learning Individual Styles of Conversational Gesture," CVPR 2019.
```

After downloading the Speech2Gesture dataset, your dataset folder should be like:
```
Gestures
├── frames_df_10_19_19.csv
├── almaram
    ├── frames
    ├── keypoints_all
    ├── keypoints_simple
    └── videos
...
└── shelly
    ├── frames
    ├── keypoints_all
    ├── keypoints_simple
    └── videos
```
2. Download the text dataset from [HERE](https://drive.google.com/file/d/1OjSJ-F9hoLOfecF5FwdCGG2Mp8fBPgGb/view?usp=sharing) and unarchive the zip file.
3. Move the `words` directory in each speaker name's directory to the corresponding speaker's directory in your dataset directory.

After this step, your dataset folder should be like:
```
Gestures
├── frames_df_10_19_19.csv
├── almaram
    ├── frames
    ├── keypoints_all
    ├── keypoints_simple
    ├── videos
    └── words
...
└── shelly
    ├── frames
    ├── keypoints_all
    ├── keypoints_simple
    ├── videos
    └── words
```
Note that the word data for speaker Jon is very little. Therefore, it should not use for model training.

4. Set up the fasttext by following the instruction [HERE](https://fasttext.cc/docs/en/support.html). Download the pre-trained model file (wiki-news-300d-1M-subword.bin) of fasttext from [HERE](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip).

## Create training and test data
* Run the script as  
```shell
python dataset.py --base_path <BASE_PATH> --speaker <SPEAKER_NAME> --wordvec_file <W2V_FILE> --dataset_type <DATASET_TYPE> --frames <FRAMES>
```

* Options
    * <BASE_PATH>: Path to dataset folder (e.g., `/path_to_your_dataset/Gestures/`)
    * <SPEAKER_NAME>: Speaker name (directory name of speaker) (e.g., `almaram`, `oliver`)
    * <W2V_FILE>: Path to the pre-trained model of `fasttext` (e.g., `/path_to_your_fasttext_dir/wiki-news-300d-1M-subword.bin/`)
    * <DATASET_TYPE>: Dataset type (`train` or `test`)
    * <FRAMES>: Number of frame for training data (we used 64 for training and 192 for test data)

* Example (Create Oliver's data)
```shell
# training data
python dataset.py --base_path <BASE_PATH> --speaker oliver --wordvec_file <W2V_FILE> --dataset_type train --frames 64

# test data
python dataset.py --base_path <BASE_PATH> --speaker oliver --wordvec_file <W2V_FILE> --dataset_type test --frames 192
```

After run the script, the direcrories containing the training or test data are created in your dataset folder. After this step, your dataset folder should be like:
```
Gestures
├── frames_df_10_19_19.csv
├── almaram
    ├── frames
    ├── keypoints_all
    ├── keypoints_simple
    ├── test-192
    ├── train-64
    ├── videos
    └── words
...
├── shelly
    ├── frames
    ├── keypoints_all
    ├── keypoints_simple
    ├── test-192
    ├── train-64
    ├── videos
    └── words
```

## Model training
* Run the script as
```shell
python train.py --outdir_path <OUT_DIR> --speaker <SPEAKER_NAME> --gpu_num <GPU> --base_path <BASE_PATH> --train_dir <TRAIN_DIR>
```

* Options
    * <OUT_DIR>: Directory for saving training result (e.g., `./out_training/`)
    * <SPEAKER_NAME>: Speaker name (directory name of speaker) (e.g., `almaram`, `oliver`)
    * <GPU>: GPU ID
    * <BASE_PATH>: Path to dataset folder (e.g., `/path_to_your_dataset/Gestures/`)
    * <TRAIN_DIR>: Directory name containing training data (e.g., `train-64`)

The experimental settings (e.g., number of epochs, loss function) can change by specifying the argument. Please see the script file of `train.py` for the details.

* Example (Training using Oliver's data)
```shell
python train.py --outdir_path ./out_training/ --speaker oliver --gpu_num 0 --base_path <BASE_PATH> --train_dir train-64
```
The resulting files will be created in `./out_training/oliver_YYYYMMDD-AAAAAA/`.

## Evaluation
* Predict the gesture motion for test data using a trained model
* Run the script as
```shell
python test.py --base_path <BASE_PATH> --test_speaker <TEST_SPEAKER> --test_dir <TEST_DIR> --model_dir <MODEL_DIR> --model_path <MODEL_PATH> --outdir_path <OUT_DIR>
```

* Options
    * <BASE_PATH>: Path to dataset folder (e.g., `/path_to_your_dataset/Gestures/`)
    * <TEST_SPEAKER>: Speaker name for testing (directory name of test speaker) (e.g., `almaram`, `oliver`)
    * <TEST_DIR>: Directory name containing test data (e.g., `test-192`)
    * <MODEL_DIR>: Directory name of trained model (e.g., `oliver_YYYYMMDD-AAAAAA`)
    * <MODEL_PATH>: Path to training result (e.g., `./out_training/`)
    * <OUT_DIR>: Directory for saving test result (e.g., `./out_test/`)

* Example (Predict the Oliver's test data using Oliver's trained model)
```shell
python test.py --base_path <BASE_PATH> --test_speaker oliver --test_dir test-192 --model_dir oliver_YYYYMMDD-AAAAAA --model_path ./out_training/ --outdir_path ./out_test/
```
The resulting files (`.npy` files for predicted motion) are created in `./out_test/oliver_by_oliver_YYYYMMDD-AAAAAA_test-192/`.

* Example (Predict the Rock's test data using Oliver's trained model)
```shell
python test.py --base_path <BASE_PATH> --test_speaker rock --test_dir test-192 --model_dir oliver_YYYYMMDD-AAAAAA --model_path ./out_training/ --outdir_path ./out_test/
```
The resulting files (`.npy` files for predicted motion) will be created in `./out_test/rock_by_oliver_YYYYMMDD-AAAAAA_test-192/`.

## Visualization
* Create gesture movie files
* Run the script as
```shell
python make_gesture_video.py --base_path <BASE_PATH> --test_out_path <TEST_OUT_PATH> --test_out_dir <TEST_OUT_DIR> --video_out_path <VIDEO_OUT_PATH>
```

* Options
    * <BASE_PATH>: Path to dataset folder (e.g., `/path_to_your_dataset/Gestures/`)
    * <TEST_OUT_PATH>: Directory path of test output (e.g., `./out_test/`)
    * <TEST_OUT_DIR>: Directory name of output gestures (e.g., `oliver_by_oliver_YYYYMMDD-AAAAAA_test-192`)
    * <VIDEO_OUT_PATH>: Directory path of output videos (e.g., `./out_video/`)

* Example
```shell
python make_gesture_video.py --base_path <BASE_PATH> --test_out_path ./out_test/ --test_out_dir oliver_by_oliver_YYYYMMDD-AAAAAA_test-192 --video_out_path ./out_video/
```

The gesture videos (side-by-side video of ground truth and text-to-gesture) will be created in `./out_video`. The left side gesture is ground truth, and the right side gesture is one generated by the text-to-gesture generation model. Also, the original videos of test intervals will be created in `./out_video/original/oliver/`.

---

## For Transformer model
If you want to use the transformer model, please use `./transformer/dataset.py`, `train_transformer.py`, and `test_transformer.py` instead of `./dataset.py`, `train.py`, and `test.py`.

### Create training and test data
The training dataset creation code `./transformer/dataset.py` creates the data including both text and audio information for model training. The created files are saved in `train-64-text-audio` instead of `train-64`, which should be used for transformer model training.

### Model training
* Example (Training using Oliver's data) with 
```shell
# Text2Gesture
python train_transformer.py --outdir_path ./out_training/ --speaker oliver --gpu_num 0 --base_path <BASE_PATH> --train_dir train-64-text-audio --modality text

# Speech2Gesture
python train_transformer.py --outdir_path ./out_training/ --speaker oliver --gpu_num 0 --base_path <BASE_PATH> --train_dir train-64-text-audio --modality audio
```

### Model training
* Example (Predict the Rock's test data using Oliver's trained model)
```shell
# Text2Gesture
python test_transformer.py --modality text --base_path <BASE_PATH> --test_speaker rock --test_dir test-192 --model_dir oliver_YYYYMMDD-AAAAAA --model_path ./out_training/ --outdir_path ./out_test/

# Speech2Gesture
python test_transformer.py --modality audio --base_path <BASE_PATH> --test_speaker rock --test_dir test-192 --model_dir oliver_YYYYMMDD-AAAAAA --model_path ./out_training/ --outdir_path ./out_test/
```
