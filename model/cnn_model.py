import torch
from torch import nn

import pfrl


class Net2(nn.Module):
    def __init__(self, action_size, dueling=False):
        super(Net2, self).__init__()
        self.action_size = action_size
        self.dueling = dueling

        self.cnn1 = nn.Sequential(nn.Conv2d(3, 100, (1, 1), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(100))
        self.cnn2 = nn.Sequential(nn.Conv2d(100, 60, (2, 2), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(60))
        self.cnn3 = nn.Sequential(nn.Conv2d(60, 50, (3, 3), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(50))
        self.cnn4 = nn.Sequential(nn.Conv2d(50, 20, (6, 6), stride=(1, 1), padding=0), nn.ReLU(), nn.BatchNorm2d(20))

        if not dueling:
            self.value_mlp = nn.Sequential(
                nn.Linear(80, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size),
            )
        else:
            self.value_mlp = nn.Sequential(
                nn.Linear(80, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.advantage_mlp = nn.Sequential(
                nn.Linear(80, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.cnn1(x)
        h = self.cnn2(h)
        h = self.cnn3(h)
        h = self.cnn4(h)

        h = h.flatten().reshape((batch_size, -1))

        if not self.dueling:
            action_values = self.value_mlp(h)
            return pfrl.action_value.DiscreteActionValue(action_values)
        else:
            # Advantage
            ya = self.advantage_mlp(h)
            mean = torch.reshape(torch.sum(ya, dim=1) / self.action_size, (batch_size, 1))
            ya, mean = torch.broadcast_tensors(ya, mean)
            ya -= mean

            # state value
            ys = self.value_mlp(h)
            ya, ys = torch.broadcast_tensors(ya, ys)
            action_values = ya + ys
            return pfrl.action_value.DiscreteActionValue(action_values)
