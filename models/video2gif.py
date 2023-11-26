import torch
import torch.nn as nn


class Video2Gif(nn.Module):
    """The Video2Gif network."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.h1 = nn.Linear(4096, 512)
        self.h2 = nn.Linear(512, 128)
        self.score = nn.Linear(128, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        if pretrained:
            self.__load_pretrained_weights('../data/features/c3d.pickle')

    def forward(self, x: torch.Tensor, extract_features: bool = False) -> torch.Tensor:
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        # print("fc6 input: ", h.sum())
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.h1(h))
        h = self.dropout(h)
        h = self.relu(self.h2(h))
        h = self.dropout(h)

        score = self.score(h)
        # print("score output: ", score)

        return score

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, extract_features=True)

    def __load_pretrained_weights(self, path):
        """Initialiaze network."""
        p_dict = torch.load(path)
        s_dict = self.state_dict()
        # print("p_dict: ", p_dict)
        # print("s_dict: ", s_dict)
        for name in p_dict:
            if name not in s_dict:
                # print("layer not match: ", name)
                continue
            s_dict[name] = p_dict[name]
            # print("layer match: ", name)
        self.load_state_dict(s_dict)


def pretrained_video2gif() -> torch.nn.Module:
    video2gif = Video2Gif(pretrained=True)
    video2gif.eval()
    for param in video2gif.parameters():
        param.requires_grad = False
    return video2gif
