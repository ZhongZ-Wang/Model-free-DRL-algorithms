import torch
import torch.nn as nn


def weights_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class QnetTwin(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QnetTwin, self).__init__()  # 调用Qnet的父类的属性及方法
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(weights_init_)

    def forward(self, state):
        action_value = self.net(state)
        return action_value

