import torch
import torch.nn as nn
from torch.distributions import Normal, Beta
import torch.nn.functional as F


class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GaussianActor_mu, self).__init__()  # 调用Qnet的父类的属性及方法
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # 自动进行参数训练

    def forward(self, state):
        s = self.net(state)
        mu = torch.sigmoid(self.mu(s))
        return mu

    def get_dist(self, state):
        mu = self.forward(state)
        log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist


class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GaussianActor_musigma, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        a = self.net(state)
        mu = F.sigmoid(self.mu(a))
        sigma = F.softplus(self.sigma(a))
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        return dist


class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(BetaActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.alpha = nn.Linear(hidden_dim, action_dim)
        self.beta = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        a = self.net(state)
        alpha = F.softplus(self.alpha(a)) + 1.0
        beta = F.softplus(self.beta(a)) + 1.0
        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, state):
        alpha, beta = self.forward(state)
        mean = (alpha) / (alpha + beta)
        return mean


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        value = self.net(state)
        return value


