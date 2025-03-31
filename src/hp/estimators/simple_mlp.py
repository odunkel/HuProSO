import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


"""N layers MLP"""


class MLPTransform(nn.Module):
    """
    A N layer NN with ReLU as activation function
    """

    def __init__(self, Ni: int, No: int, Nh: int = 16, num_hid_layers: int = 1):
        super(MLPTransform, self).__init__()

        layers = []
        self.fc_first = nn.Linear(Ni, Nh)
        for i in range(num_hid_layers):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(Nh, Nh))
        self.relu_last = nn.LeakyReLU()
        self.fc_last = nn.Linear(Nh, No)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.fc_first(x)
        x0 = x.clone()
        for layer in self.layers:
            x = layer(x)
        x = self.relu_last(x0 + x)
        return self.fc_last(x)
