import matplotlib
import subprocess
import os
import numpy as np
from matplotlib import cm, pyplot as plt
from PIL import Image

matplotlib.use("Agg")


# Keypoint consts
BASE_KEYPOINT = [0]
RIGHT_BODY_KEYPOINTS = [1, 2, 3, 28]
LEFT_BODY_KEYPOINTS = [4, 5, 6, 7]
LEFT_HAND_KEYPOINTS = lambda x: [7] + [8 + (x * 4) + j for j in range(4)]
RIGHT_HAND_KEYPOINTS = lambda x: [28] + [29 + (x * 4) + j for j in range(4)]
LINE_WIDTH = 1.5
MARKER_SIZE = 1.5


def plot_body_right_keypoints(keypoints, color, alpha=None, line_width=LINE_WIDTH):
    _keypoints = np.array(BASE_KEYPOINT + RIGHT_BODY_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha, color=color)


def plot_body_left_keypoints(keypoints, color, alpha=None, line_width=LINE_WIDTH):
    _keypoints = np.array(BASE_KEYPOINT + LEFT_BODY_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha, color=color)


def plot_left_hand_keypoints(keypoints, color, alpha=None, line_width=LINE_WIDTH):
    for i in range(5):
        _keypoints = np.array(LEFT_HAND_KEYPOINTS(i))
        plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha, color=color)


def plot_right_hand_keypoints(keypoints, color, alpha=None, line_width=LINE_WIDTH):
    for i in range(5):
        _keypoints = np.array(RIGHT_HAND_KEYPOINTS(i))
        plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha, color=color)


def plot_face(keypoints, color, alpha=None, marker_size=MARKER_SIZE):
    plt.plot(keypoints[0], keypoints[1], color, marker='.', lw=0, markersize=marker_size, alpha=alpha)


def draw_poses(img, frame_body_kpts, frame_face_kpts, img_size, output=None, show=None, title=None, sub_size=None, color=None):
    # Number of persons to draw
    persons = len(frame_body_kpts)
    
    plt.close('all')
    fig = plt.figure(figsize=(persons * 2, 1 * 2), dpi=200)

    # Draw title
    if title is not None:
        plt.title(title)

    if img is not None:
        img_ = Image.open(img)
    else:
        img_ = Image.new(mode='RGB', size=img_size, color='white')
    plt.imshow(img_, alpha=0.5)

    if color is None:
        color = ['dodgerblue'] * persons

    for i in range(len(frame_body_kpts)):
        ax = plt.subplot(1, persons, i+1)

        plot_body_right_keypoints(frame_body_kpts[i], color[i])
        plot_body_left_keypoints(frame_body_kpts[i], color[i])
        plot_left_hand_keypoints(frame_body_kpts[i], color[i])
        plot_right_hand_keypoints(frame_body_kpts[i], color[i])
        if frame_face_kpts is not None:
            plot_face(frame_face_kpts[i], color[i])
        # Plotting size specification
        ax.set_xlim(sub_size[0], sub_size[1])
        ax.set_ylim(sub_size[1], sub_size[0])
        # Remove axis
        plt.axis('off')

    # Remove axis
    plt.axis('off')

    if show:
        plt.show()
    if output is not None:
        plt.savefig(output)


def create_mute_video_from_images(output_fn, temp_folder):
    """
    :param output_fn: output video file name
    :param temp_folder: contains images in the format 0001.jpg, 0002.jpg....
    :return:
    """
    subprocess.call('ffmpeg -r 30000/2002 -f image2 -i "%s" -r 30000/1001 "%s" -y' % (
        os.path.join(temp_folder, '%04d.jpg'), output_fn), shell=True)


def create_voice_video_from_voice_and_video(audio_input_path, input_video_path, output_video_path):
    subprocess.call('ffmpeg -i "%s" -i "%s" -strict -2 "%s" -y' % (audio_input_path, input_video_path,
                                                                   output_video_path), shell=True)


def save_video(body_kpts, face_kpts, voice_path, output_fn, temp_folder, delete_tmp=False, color=None,
               img_size=(720, 480), img=None):
    # Create temporary directory
    os.makedirs(temp_folder, exist_ok=True)
    # Number of frames
    frames = len(body_kpts[0])
    # Temporary file name pattern
    output_fn_pattern = os.path.join(temp_folder, '%04d.jpg')
    
    # Size for drawing
    if face_kpts is None:
        sub_max = np.max(body_kpts)
        sub_min = np.min(body_kpts)
    else:
        sub_max = max([np.max(body_kpts), np.max(face_kpts)])
        sub_min = min([np.min(body_kpts), np.min(face_kpts)])

    # Drawing per frame
    for j in range(frames):
        frame_body_kpts = body_kpts[:, j]
        if face_kpts is None:
            frame_face_kpts = None
        else:
            frame_face_kpts = face_kpts[:, j]
        draw_poses(None, frame_body_kpts, frame_face_kpts, output=output_fn_pattern % j, show=False, color=color,
                   img_size=img_size, sub_size=[sub_min, sub_max])
        plt.close()
    
    # Create mute video
    create_mute_video_from_images(output_fn, temp_folder)
    # Create video with voice
    if voice_path is not None:
        create_voice_video_from_voice_and_video(voice_path, output_fn, str(output_fn)[:-4] + '-voice.mp4')
        subprocess.call('rm "%s"' % output_fn, shell=True)
    # Delete temporary files
    if delete_tmp:
        subprocess.call('rm -R "%s"' % temp_folder, shell=True)
