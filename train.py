import torch
from datasets.datasets import RankDataset
from torch.utils.data import DataLoader
from features.c3d import pretrained_c3d
from models.varnet import VARNet
from models.varnet import loss_func
from tools import video_utils
import os
import json
from datasets.static import TVSUM_SPLITS
from tools import data_preprocess


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE being used:", DEVICE)

epochs = 10  # Number of epochs for training
batch_size = 1
lr = 1e-3  # Learning rate

DATASET = 'tvsum'
DATA_BASE_DIR = 'data/'
TVSUM_DATA_DIR = DATA_BASE_DIR + 'tvsum'
SAVE_DIR = DATA_BASE_DIR + 'save'
SPLITS = TVSUM_SPLITS
VIDEO_TYPE = "GA"


def train():
    global idx
    # 解析标注数据json文件
    anno_file = os.path.join(TVSUM_DATA_DIR, "data/tvsum_anno.json")
    with open(anno_file, 'r') as f:
        anno_data = json.load(f)
#     print(anno_data)
    train_videos = SPLITS[VIDEO_TYPE]["train"]
    val_videos = SPLITS[VIDEO_TYPE]["val"]
    # 初始化模型
    feature_model = pretrained_c3d().to(DEVICE)
    model = VARNet().to(DEVICE)
    # 初始化优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), eps=10 ** -8)
    # 初始化存储模型的目录
    save_dir = os.path.join(SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
        # train
        print("epoch {} start train".format(epoch))
        model.train()
        for video_name in train_videos:
            # video_name = "-esJrBWj2d8"
            print("epoch {} train {} start".format(epoch, video_name))
            scores = anno_data[video_name]["score_mean"]
            video_indexes = [i for i in range(len(scores))]
            # 初始化数据加载器
            data_set = RankDataset(video_indexes, scores)
            data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
            # loss_train = 0
            for idx, (xi, yi, xj, yj) in enumerate(data_loader):
                print("pick video({}) {} and {}, score is {} and {} to train".format(video_name, xi, xj, yi, yj))
                # forward
                videoi = os.path.join(TVSUM_DATA_DIR, "video_clip/{}/{}.mp4".format(video_name, xi[0]))
                videoj = os.path.join(TVSUM_DATA_DIR, "video_clip/{}/{}.mp4".format(video_name, xj[0]))
                score = model.forward(feature_model.extract_features(video_utils.get_video_5d_tensor(videoi).to(DEVICE)).to(DEVICE),
                                  feature_model.extract_features(video_utils.get_video_5d_tensor(videoj).to(DEVICE)).to(DEVICE))
                loss = loss_func(score, yi, yj)
                print("loss is ", loss)
                # loss_train += loss.cpu().detach().numpy()
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#                 break
            # loss_train /= (idx + 1)
            # print("loss_train is ", loss_train)
            model_file_path = os.path.join(save_dir, "varnet_{}_{}_{}_{}.ckpt".format(DATASET, VIDEO_TYPE, epoch, video_name))
            print("save {}".format(model_file_path))
            torch.save(model.state_dict(), model_file_path)
#             break

        # valid
        print("epoch {} start valid".format(epoch))
        model.eval()
        for video_name in val_videos:
            print("epoch {} valid {} start".format(epoch, video_name))
            scores = anno_data[video_name]["score_mean"]
            sorted_indexes = anno_data[video_name]["index"]  # to calculate mAP
            predict_scores = []
            for i in range(len(scores)):
                video = os.path.join(TVSUM_DATA_DIR, "video_clip/{}/{}.mp4".format(video_name, i))
                predict_scores.append(model.get_score(feature_model.extract_features(video_utils.get_video_5d_tensor(video).to(DEVICE)).to(DEVICE)))
            predict_sorted_indexes = data_preprocess.get_sorted_index(predict_scores)
            ap = data_preprocess.get_average_precision(sorted_indexes, predict_sorted_indexes)
            print("epoch {} valid ap {}".format(epoch, ap))
#             break
        print("epoch {} end".format(epoch))
#         break


if __name__ == '__main__':
    train()
