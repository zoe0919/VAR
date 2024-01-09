import torch
import torch.nn as nn


class VARNet(nn.Module):
    """The VAR network."""

    def __init__(self) -> None:
        super().__init__()
        self.h1 = nn.Linear(4096, 512)
        self.h2 = nn.Linear(512, 128)
        self.score = nn.Linear(128, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.h1(x))
        h = self.dropout(h)
        h = self.relu(self.h2(h))
        h = self.dropout(h)

        score = self.score(h)
        # print("score output: ", score)

        return score

    def forward(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        score_i = self.get_score(xi)  # 预测x1得分
        score_j = self.get_score(xj)  # 预测x2得分
        pred = self.sigmoid(score_i - score_j)  # 归一化
        return pred


# from https://zhuanlan.zhihu.com/p/435989798?utm_id=0
def loss_func(score, yi, yj):
    sij = torch.zeros_like(score)
    sij[yi > yj] = 1
    sij[yi == yj] = 0
    sij[yi < yj] = -1

    pij = 0.5 * (1 + sij)
    func = nn.BCELoss()
    loss = func(score, pij)
    return loss


def pretrained_varnet() -> torch.nn.Module:
    varnet = VARNet(pretrained=True)
    varnet.eval()
    for param in varnet.parameters():
        param.requires_grad = False
    return varnet
