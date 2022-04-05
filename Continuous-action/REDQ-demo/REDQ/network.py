import torch
import torch.nn as nn
from torch.distributions import Normal


# def weights_init_(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         torch.nn.init.xavier_uniform_(m.weight)
#         torch.nn.init.constant_(m.bias, 0)
Log_std_max = 2
Log_std_min = -20


def weights_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.uniform_(m.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(m.bias, a=-3e-3, b=3e-3)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()  # 调用Qnet的父类的属性及方法
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        action = self.net(state)
        mu = self.mu(action)
        log_std = self.log_std(action)
        log_std = torch.clamp(log_std, Log_std_min, Log_std_max)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        value = self.net(torch.cat([state, action], 1))
        return value


