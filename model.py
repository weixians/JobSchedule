import torch
from torch import nn

import pfrl


class Net2(nn.Module):
    def __init__(self, action_size):
        super(Net2, self).__init__()
        self.action_size = action_size

        self.value_cnn = nn.Sequential(
            nn.Conv2d(3, 100, (1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(100),
            nn.Conv2d(100, 60, (2, 2), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 50, (3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(50, 20, (6, 6), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )
        self.value_mlp = nn.Sequential(nn.Linear(41472, 512), nn.ReLU(), nn.Linear(512, action_size))

    def forward(self, x):
        out1 = self.value_cnn(x)
        out2 = self.value_mlp(out1)
        return pfrl.action_value.DiscreteActionValue(out2)
