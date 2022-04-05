import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_atoms=51, vmax=200, vmin=0):
        super(Qnet, self).__init__()  # 调用Qnet的父类的属性及方法
        self.n_atoms = n_atoms
        self.vmax = vmax
        self.vmin = vmin
        # self.support = torch.arange(self.vmin, self.vmax + self.delta_z, self.delta_z)
        self.support = torch.linspace(self.vmin, self.vmax, self.n_atoms)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * self.n_atoms),
        )
        self.apply(weights_init_)

    def forward(self, state):
        batch_size = state.size()[0]
        state = torch.as_tensor(state, dtype=torch.float32)
        dist = self.net(state).view(batch_size, -1, self.n_atoms)
        dist = F.softmax(dist, dim=-1)
        dist = dist.clamp(min=1e-3)  # 避免nan值
        q = torch.sum(dist * self.support, dim=2)
        return dist, q

