import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class A_net(nn.Module):
    def __init__(self, state_n, action_n, width, a_range):
        super(A_net, self).__init__()
        self.A = nn.Sequential(
            layer_init(nn.Linear(state_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, action_n)),
            nn.Tanh()
        )
        self.range = a_range[-1]

    def forward(self, s):
        a = self.A(s)  # -1:1
        action = a * self.range
        return action


class C_net(nn.Module):
    def __init__(self, state_n, action_n, width):
        super(C_net, self).__init__()
        self.C = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, width)),
            nn.ReLU(),
            layer_init(nn.Linear(width, 1)),
            nn.Identity()
        )

    def forward(self, s, a):
        return self.C(torch.cat([s, a], -1))
