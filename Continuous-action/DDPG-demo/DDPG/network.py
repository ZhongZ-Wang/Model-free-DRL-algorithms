import torch
import torch.nn as nn


# def weights_init_(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         torch.nn.init.xavier_uniform_(m.weight)
#         torch.nn.init.constant_(m.bias, 0)


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
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(weights_init_)

    def forward(self, state):
        # state = torch.as_tensor(state, dtype=torch.float32)
        action = self.net(state).tanh()
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        value = self.net(torch.cat([state, action], 1))
        return value


