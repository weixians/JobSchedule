import torch
from torch import nn

import pfrl


class Net2(nn.Module):
    def __init__(self, action_size):
        super(Net2, self).__init__()
        self.action_size = action_size

        self.cnn1 = nn.Sequential(nn.Conv2d(3, 100, (1, 1), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(100))
        self.cnn2 = nn.Sequential(nn.Conv2d(100, 60, (2, 2), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(60))
        self.cnn3 = nn.Sequential(nn.Conv2d(60, 50, (3, 3), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(50))
        self.cnn4 = nn.Sequential(nn.Conv2d(50, 20, (6, 6), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(20))

        self.value_mlp = nn.Sequential(nn.Linear(80, 512), nn.ReLU(), nn.Linear(512, action_size))

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.cnn1(x)
        h = self.cnn2(h)
        h = self.cnn3(h)
        h = self.cnn4(h)

        h = h.flatten().reshape((batch_size, -1))

        out2 = self.value_mlp(h)
        return pfrl.action_value.DiscreteActionValue(out2)
