import numpy as np
import torch
import torch.nn.functional as Func
import copy
from Net_con import A_net, C_net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DDPG_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.Actor = A_net(self.state_dim, self.action_dim, self.net_width, self.a_range).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)
        self.A_target = copy.deepcopy(self.Actor)

        self.Critic = C_net(self.state_dim, self.action_dim, self.net_width).to(device)
        self.C_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.c_lr)
        self.C_target = copy.deepcopy(self.Critic)

    def action_select(self, s, iseval):
        with torch.no_grad():
            state = torch.FloatTensor(s).view(1, -1).to(device)
            a = self.Actor(state).cpu().numpy().squeeze(0)
            if iseval:
                action = a
            else:
                noise = np.random.normal(0, self.noise * self.a_range[1], self.action_dim)
                action = np.clip(noise + a, self.a_range[0], self.a_range[1])
            return action

    def train(self, Replay):
        s, a, r, s_, done = Replay.sample(self.batch_size)
        with torch.no_grad():
            a_ = self.A_target(s_)
            Q_target = r + self.gamma * self.C_target(s_, a_) * (~done)

        Q = self.Critic(s, a)
        C_loss = Func.mse_loss(Q_target, Q)
        self.C_optimizer.zero_grad()
        C_loss.backward()
        self.C_optimizer.step()

        A_loss = torch.mean(-self.Critic(s, self.Actor(s)))
        self.A_optimizer.zero_grad()
        A_loss.backward()
        self.A_optimizer.step()

        with torch.no_grad():
            for p, p_t in zip(self.Actor.parameters(), self.A_target.parameters()):
                p_t.data.copy_(self.tua * p.data + (1 - self.tua) * p_t.data)

            for p, p_t in zip(self.Critic.parameters(), self.C_target.parameters()):
                p_t.data.copy_(self.tua * p.data + (1 - self.tua) * p_t.data)

    def save(self, EnvName, timestep):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor{}.pth".format(EnvName, timestep))
        torch.save(self.Critic.state_dict(), "./model/{}_Critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.Actor.load_state_dict(torch.load("./model/{}_Actor{}.pth".format(EnvName, timestep), map_location=device))
        self.Critic.load_state_dict(
            torch.load("./model/{}_Critic{}.pth".format(EnvName, timestep), map_location=device))


class Buffer_replay(object):
    def __init__(self, action_n, state_n, max_size):
        self.max_size = int(max_size)
        self.Ind = int(0)
        self.s = np.zeros((self.max_size, state_n), dtype=np.float32)
        self.s_ = copy.deepcopy(self.s)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.a = np.zeros((self.max_size, action_n), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.bool_)
        self.dw = copy.deepcopy(self.done)
        self.size = int(0)

    def add(self, s, a, r, s_, done, dw):
        Ind = self.Ind
        self.a[Ind] = a
        self.s[Ind] = s
        self.s_[Ind] = s_
        self.r[Ind] = r
        self.done[Ind] = done
        self.dw[Ind] = dw
        self.Ind = (self.Ind + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, b_size):
        Ind = np.random.choice(self.size, b_size, replace=False)
        return (
            torch.FloatTensor(self.s[Ind]).to(device),
            torch.FloatTensor(self.a[Ind]).to(device),
            torch.FloatTensor(self.r[Ind]).to(device),
            torch.FloatTensor(self.s_[Ind]).to(device),
            torch.BoolTensor(self.done[Ind]).to(device),
        )
