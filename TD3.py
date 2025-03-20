import torch
import torch.nn.functional as F
import numpy as np
import copy
from Net_con import A_net, C_net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TD3_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.Actor = A_net(self.action_dim, self.state_dim, self.net_width, self.a_range).to(device)
        self.A_target = copy.deepcopy(self.Actor)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)
        self.Critic1 = C_net(self.action_dim, self.state_dim, self.net_width).to(device)
        self.Critic2 = C_net(self.action_dim, self.state_dim, self.net_width).to(device)
        self.C1_optimizer = torch.optim.Adam(self.Critic1.parameters(), lr=self.c_lr)
        self.C2_optimizer = torch.optim.Adam(self.Critic2.parameters(), lr=self.c_lr)

        self.C1_target = copy.deepcopy(self.Critic1)
        self.C2_target = copy.deepcopy(self.Critic2)

        self.a_std = self.exp_noise * self.a_range[-1]

        self.policy_noise = 0.2 * self.a_range[-1]
        self.noise_clip = 0.5 * self.a_range[-1]

        self.delay_counter = 0

    def action_select(self, s, iseval):
        with torch.no_grad():
            state = torch.FloatTensor(s).view(1, -1).to(device)
            a = self.Actor(state).cpu().numpy().squeeze(0)
            if iseval:
                action = a
            else:
                noise = np.random.normal(0, self.a_std, self.action_dim)
                action = np.clip(a + noise, self.a_range[0], self.a_range[1])
            return action

    def train(self, Replay):
        self.delay_counter += 1
        s, a, r, s_, done = Replay.sample(self.batch_size)
        with torch.no_grad():
            a_next = self.A_target(s_)
            noise_a = torch.clip(torch.randn_like(a) * self.policy_noise, -self.noise_clip, self.noise_clip)
            a_next_n = torch.clip(noise_a + a_next, self.a_range[0], self.a_range[-1])
            Q1_ = self.C1_target(s_, a_next_n)
            Q2_ = self.C2_target(s_, a_next_n)
            Q_ = torch.min(Q1_, Q2_)
            Q_target = r + (~done) * self.gamma * Q_

        Q1 = self.Critic1(s, a)
        Q2 = self.Critic2(s, a)

        C1_loss = F.mse_loss(Q_target, Q1)
        self.C1_optimizer.zero_grad()
        C1_loss.backward()
        self.C1_optimizer.step()

        C2_loss = F.mse_loss(Q_target, Q2)
        self.C2_optimizer.zero_grad()
        C2_loss.backward()
        self.C2_optimizer.step()

        if self.delay_counter % self.policy_delay_freq == 0:
            A_loss = -self.Critic1(s, self.Actor(s)).mean()
            self.A_optimizer.zero_grad()
            A_loss.backward()
            self.A_optimizer.step()

            for p, p_target in zip(self.Actor.parameters(), self.A_target.parameters()):
                p_target.data.copy_(self.tua * p.data + (1 - self.tua) * p_target.data)

            for p, p_target in zip(self.Critic1.parameters(), self.C1_target.parameters()):
                p_target.data.copy_(self.tua * p.data + (1 - self.tua) * p_target.data)

            for p, p_target in zip(self.Critic2.parameters(), self.C2_target.parameters()):
                p_target.data.copy_(self.tua * p.data + (1 - self.tua) * p_target.data)

    def load(self, Env_name, Index):
        self.Actor.state_dict(torch.load("./model/{}_Actor{}.pth".format(Env_name, Index)), map_location=device)
        self.Critic1.state_dict(torch.load("./model/{}_Critic1{}.pth".format(Env_name, Index)), map_location=device)
        self.Critic2.state_dict(torch.load("./model/{}_Critic2{}.pth".format(Env_name, Index)), map_location=device)
        self.C1_target = copy.deepcopy(self.Critic1)
        self.C2_target = copy.deepcopy(self.Critic2)

    def save(self, Env_name, Index):
        torch.save(self.Actor.state_dict(), "./model/{}_Actor{}.pth".format(Env_name, Index))
        torch.save(self.Critic1.state_dict(), "./model/{}_Critic1{}.pth".format(Env_name, Index))
        torch.save(self.Critic2.state_dict(), "./model/{}_Critic2{}.pth".format(Env_name, Index))


class Buffer_Replay(object):
    def __init__(self, state_n, action_n, max_size):
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
