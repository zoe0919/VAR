from torch.utils.data import Dataset
from itertools import combinations


# from https://zhuanlan.zhihu.com/p/435989798?utm_id=0
class RankDataset(Dataset):
    def __init__(self, x, y):
        self.x = x  # 数据
        self.y = y  # 标记值
        self.pair = combinations(range(len(self.x)), 2)

    def __len__(self):
        return int(len(self.x) * (len(self.x) - 1) / 2)

    def __getitem__(self, index):
        i, j = next(self.pair)
        xi = self.x[i]
        yi = self.y[i]
        xj = self.x[j]
        yj = self.y[j]
        return xi, yi, xj, yj
