import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.Tensor(output_dim, input_dim))

        self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim))
        self.register_buffer('bias_epsilon', torch.Tensor(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.input_dim))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.output_dim))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Qnet, self).__init__()  # 调用Qnet的父类的属性及方法
        self.linear = nn.Linear(state_dim, hidden_dim)
        self.noiselinear1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noiselinear2 = NoisyLinear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        x = F.relu(self.linear(state))
        x = F.relu(self.noiselinear1(x))
        x = self.noiselinear2(x)
        return x

    def reset_noise(self,):
        self.noiselinear1.reset_noise()
        self.noiselinear2.reset_noise()