from moviepy.editor import *
import numpy as np
from PIL import Image
import torch


def crop_video_to_video(input_video_path, output_video_path, start_time, end_time):
    """
    从视频文件中截取部分保存为另一个视频文件
    :param input_video_path:
    :param output_video_path:
    :param start_time:
    :param end_time:
    :return:
    """
    video = VideoFileClip(input_video_path)
    clipped_video = video.subclip(start_time, end_time)
    final_video = CompositeVideoClip([clipped_video])
    final_video.write_videofile(output_video_path)


def crop_video_to_gif(input_video_path, output_video_path, start_time, end_time):
    """
    从视频文件中截取部分保存为gif文件
    :param input_video_path:
    :param output_video_path:
    :param start_time:
    :param end_time:
    :return:
    """
    video = VideoFileClip(input_video_path)
    clipped_video = video.subclip(start_time, end_time)
    final_video = CompositeVideoClip([clipped_video])
    final_video.write_gif(output_video_path)


def crop_video_to_sequence(input_video_path, output_format, start_time, end_time):
    """
    从视频文件中截取部分保存为一系列图片
    :param input_video_path:
    :param output_format:
    :param start_time:
    :param end_time:
    :return:
    """
    video = VideoFileClip(input_video_path)
    clipped_video = video.subclip(start_time, end_time)
    final_video = CompositeVideoClip([clipped_video])
    final_video.write_images_sequence(output_format)


def get_video_size(input_video_path):
    """
    获取视频分辨率
    :param input_video_path:
    :return:
    """
    return VideoFileClip(input_video_path).size


def get_video_duration(input_video_path):
    """
    获取视频时长
    :param input_video_path:
    :return:
    """
    return VideoFileClip(input_video_path).duration


def get_video_fps(input_video_path):
    """
        获取视频fps
        :param input_video_path:
        :return:
        """
    return VideoFileClip(input_video_path).fps


def get_video_frames_count(input_video_path):
    """
    获取视频帧数
    :param input_video_path:
    :return:
    """
    video = VideoFileClip(input_video_path)
    return video.fps * video.duration


def get_video_5d_tensor(video_path, clip_count=16, height_size=112, width_size=112):
    """
    将视频片段转换为可以输入给C3D和I3D的五维数据类型(B, C, T, H, W)
    :param video_path: 视频路径
    :param clip_count:
    :param height_size:
    :param width_size:
    :return:
    """
    video = VideoFileClip(video_path)
    frames_count = video.fps * video.duration
    step = int(frames_count / clip_count)
    if step == 0:
        step = 1
    frames = np.empty((clip_count, height_size, width_size, 3), np.dtype('float32'))
    frames_idx = 0
    for frame_idx, f in enumerate(video.iter_frames()):
        if frame_idx % step == 0:
            image = Image.fromarray(f)
            image = image.resize((height_size, width_size))
            image_numpy = np.array(image).astype(np.float64)
            frames[frames_idx] = image_numpy
            frames_idx += 1
        if frames_idx == clip_count:
            break
    for i in range(frames_idx, clip_count):
        frames[i] = frames[frames_idx - 1]
    frames = frames.transpose((3, 0, 1, 2))
    frames_torch = torch.from_numpy(frames)
    frames_torch = frames_torch.unsqueeze(0)
    return frames_torch

