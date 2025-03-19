import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class A_net(nn.Module):
    def __init__(self, action_n, state_n, net_width, a_range):
        super(A_net, self).__init__()
        self.A = nn.Sequential(
            layer_init(nn.Linear(state_n, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, action_n)),
            nn.Tanh()
        )
        self.a_max = a_range[-1]

    def forward(self, s):
        a = self.A(s)  # -1:1
        return a * self.a_max


class C_net(nn.Module):
    def __init__(self, action_n, state_n, net_width):
        super(C_net, self).__init__()
        self.C = nn.Sequential(
            layer_init(nn.Linear(state_n + action_n, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, net_width)),
            nn.ReLU(),
            layer_init(nn.Linear(net_width, 1)),
            nn.Identity()
        )

    def forward(self, s, a):
        Input = torch.cat([s, a], -1)
        Q = self.C(Input)
        return Q


