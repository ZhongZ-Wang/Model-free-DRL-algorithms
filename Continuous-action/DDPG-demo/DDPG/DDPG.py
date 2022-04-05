import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from network import Actor, Critic
import os, copy, random, gym


def hard_update(Q_target, Q):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data.copy_(param.data)


def soft_update(Q_target, Q, tau):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data = (1 - tau) * target_param.data + tau * param.data


class OU_noise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(DDPG, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.actor = Actor(state_dim, action_dim, args.hidden_size).to(device=self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(state_dim, action_dim, args.hidden_size)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

        self.noise = OU_noise(self.action_dim)

        self.gamma = args.gamma
        self.tau = args.tau
        self.target_update_freq = args.target_update_freq

    def select_action(self, state, is_test=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().flatten().numpy()
        if is_test:
            noise = self.noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update_parameter(self, buffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_batch, done_batch = buffer.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(device=self.device)
        action_batch = torch.FloatTensor(action_batch).to(device=self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(device=self.device).unsqueeze(1)  # 扩展维度，行变为列
        next_batch = torch.FloatTensor(next_batch).to(device=self.device)
        done_batch = torch.FloatTensor(done_batch).to(device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.target_actor(next_batch)
            target_Q = self.target_critic(next_batch, next_action) # max 返回两个值，一个为value, 一个为index
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q

        q_value = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(q_value, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - self.critic(state_batch, self.actor(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if updates % self.target_update_freq == 0:
            soft_update(self.target_critic, self.critic, self.tau)
            soft_update(self.target_actor, self.actor, self.tau)

        return critic_loss.item(), actor_loss.item(), target_Q.mean(), q_value.mean()

    def save_model(self, env_name, path=None):
        if not os.path.exists('../checkpoints/'):
            os.makedirs('../checkpoints/')
        if path is None:
            path = '../checkpoints/DDPG_checkpoint_{}'.format(env_name)
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'target_actor_state_dict': self.target_actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'target_critic_state_dict': self.target_critic.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict()}, path)

    def load_model(self, path, evaluate=False):
        if path is not None:
            checkpoint = torch.load(path)
            self.Qnet.load_state_dict(checkpoint['Qnet_state_dict'])
            self.target_Qnet.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        if evaluate:
            self.Qnet.eval()
            self.target_Qnet.eval()
        else:
            self.Qnet.train()
            self.target_Qnet.train()


class ActionNormalizer(gym.ActionWrapper):  # 写法上借鉴
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action