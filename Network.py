import math
import torch
from torch import nn
import gymnasium as gym
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, (3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, stride=(2, 2))
        self.relu2 = nn.ReLU()
        self.skip = nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, stride=(2, 2))

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        skip = self.skip(x)
        return (out + skip) / math.sqrt(2)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.res1 = ResBlock(3, 8)
        self.res2 = ResBlock(8, 16)
        self.res3 = ResBlock(16, 32)
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        batch, _, _, _ = state.shape
        x = self.res3(self.res2(self.res1(state))).view(batch, -1)
        ret = nn.functional.softmax(self.fc(x), dtype=torch.double, dim=1)
        return ret


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.res1 = ResBlock(3, 8)
        self.res2 = ResBlock(8, 16)
        self.res3 = ResBlock(16, 32)
        self.fc = nn.Linear(state_dim, 1)

    def forward(self, state):
        batch, _, _, _ = state.shape
        x = self.res3(self.res2(self.res1(state))).view(batch, -1)
        ret = self.fc(x)
        return ret


if __name__ == '__main__':
    res1 = ResBlock(3, 8)
    res2 = ResBlock(8, 16)
    res3 = ResBlock(16, 32)
    res4 = ResBlock(32, 64)
    res5 = ResBlock(64, 108)
    sample = torch.randn(1, 3, 1080, 1920)
    x = res5(res4(res3(res2(res1(sample))))).view(1, -1)
    print(x.shape)