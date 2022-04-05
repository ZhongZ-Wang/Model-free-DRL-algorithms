import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def weights_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()  # 调用Qnet的父类的属性及方法
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # noise = torch.randn_like(mean, requires_grad=True)
        # action = (mean + std * noise).tanh()
        #
        # log_prob = log_std + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        # log_prob = log_prob + (-action.pow(2) + 1.00001).log()
        # log_prob = log_prob.sum(1, keepdim=True)

        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean)
        return action, log_prob, torch.normal(mean, std).tanh()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        value1 = self.net1(torch.cat([state, action], 1))
        value2 = self.net2(torch.cat([state, action], 1))
        return value1, value2