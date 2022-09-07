import numpy as np
import torch
from torch import nn

import pfrl


class NetMLP(nn.Module):
    def __init__(self, input_dim, action_size):
        super(NetMLP, self).__init__()
        self.action_size = action_size

        input_size = np.product(input_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))

        out2 = self.value_mlp(x)
        return pfrl.action_value.DiscreteActionValue(out2)
