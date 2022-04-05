import torch
import torch.nn as nn


def weights_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class QnetDueling(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QnetDueling, self).__init__()  # 调用Qnet的父类的属性及方法
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.adv = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        feature = self.feature_layer(state)
        v = self.value(feature)
        adv = self.adv(feature)
        q = v + adv - adv.mean(dim=-1, keepdim=True)
        return q

