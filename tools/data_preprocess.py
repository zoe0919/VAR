import os
import csv
import sys
import video_utils
import math
import json
from tqdm import tqdm
import numpy as np


# indexes start from 0
def get_average_precision(ground_truth_indexes, predict_indexes):
    gt_arr = np.array(ground_truth_indexes)
    gt_arr = np.add(gt_arr, 1)
    pre_arr = np.array(predict_indexes)
    pre_arr = np.add(pre_arr, 1)
    temp = np.divide(gt_arr, pre_arr)
    return np.sum(temp) / len(temp)


# 得到原本序号(i)和排序(从大到小)后序号(output_index[i])的对应关系
def get_sorted_index(array):
    index = [i for i in range(len(array))]
    sorted_index = sorted(index, key=lambda x: array[x], reverse=True)
    output_index = [0] * len(sorted_index)
    for i, ind in enumerate(sorted_index):
        output_index[sorted_index[i]] = i
    return output_index


# tvsum数据集标注数据预处理
def tvsum_anno_preprocess(path):
    # 2s一个片段，一个片段一个评分
    my_anno = {}  # 最终生成的标注数据
    # 读取视频列表
    video_path = os.path.join(path, "video")
    files = os.listdir(video_path)
    # 解析fps、duration、frames
    for file in tqdm(files):
        # print("parse file: ", file)
        video_file = os.path.join(video_path, file)
        fps = video_utils.get_video_fps(video_file)
        duration = video_utils.get_video_duration(video_file)
        frames = fps * duration
        video_info = {"fps": fps, "duration": duration, "frames": frames}
        my_anno[os.path.splitext(file)[0]] = video_info
    # 读取评分数据并整理
    anno_path = os.path.join(path, "data/ydata-tvsum50-anno.tsv")
    with open(anno_path, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(tqdm(reader)):
            # print("parse i: ", i, "file: ", row[0])
            # 解析domain
            my_anno[row[0]]["domain"] = row[1]
            # 计算视频片段分数
            step = math.ceil(my_anno[row[0]]["fps"] * 2)
            all_scores = row[2].split(',')
            # 计算每2s的片段分数值
            scores = []
            for j, score in enumerate(all_scores):
                if j % step == 0:
                    scores.append(int(score))
            # 计算分数平均值
            if i % 20 == 0:
                my_anno[row[0]]["score_mean"] = scores
            else:
                temp_list = []
                for n, item in enumerate(my_anno[row[0]]["score_mean"]):
                    if i % 20 == 19:
                        temp_list.append((item + scores[n]) / 20)  # 计算平均值
                    else:
                        temp_list.append(item + scores[n])  # 计算分数之和
                my_anno[row[0]]["score_mean"] = temp_list
                if i % 20 == 19:
                    my_anno[row[0]]["index"] = get_sorted_index(temp_list)
    # 将my_anno写入json文件
    my_anno_path = os.path.join(path, "data/tvsum_anno.json")
    with open(my_anno_path, "w") as file:
        json.dump(my_anno, file)
    print("write my_anno json file down!")
    sys.exit()


# tvsum数据集视频预处理
def tvsum_video_preprocess(path):
    # 读取视频列表
    video_path = os.path.join(path, "video")
    files = os.listdir(video_path)
    # 创建视频片段目录
    video_clip_path = os.path.join(path, "video_clip")
    os.makedirs(video_clip_path, exist_ok=True)
    # 裁剪视频片段
    for file in files:
        video_file = os.path.join(video_path, file)
        duration = video_utils.get_video_duration(video_file)
        clip_path = os.path.join(video_clip_path, os.path.splitext(file)[0])
        os.makedirs(clip_path, exist_ok=True)
        for i in range(0, math.ceil(duration), 2):
            output_video_path = os.path.join(clip_path, "{:.0f}.mp4".format(i / 2))
            end = (i + 2) if (i + 2) < duration else duration
            video_utils.crop_video_to_video(video_file, output_video_path, i, end)
        # break


if __name__ == '__main__':
    tvsum_path = "../data/tvsum"
    # 输出重新整理的标注数据和视频片段
    tvsum_anno_preprocess(tvsum_path)
    tvsum_video_preprocess(tvsum_path)
