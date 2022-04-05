import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from network import Actor, Critic
import os, copy, gym


def hard_update(Q_target, Q):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data.copy_(param.data)


def soft_update(Q_target, Q, tau):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data = (1 - tau) * target_param.data + tau * param.data


class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(TD3, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.actor = Actor(self.state_dim, self.action_dim, args.hidden_size).to(device=self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(self.state_dim, self.action_dim, args.hidden_size)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.policy_freq = args.policy_freq
        self.noise_clip = args.noise_clip
        self.max_action = 1

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().flatten()
        return action

    def update_parameter(self, buffer, batch_size):
        self.total_it += 1

        state, action, reward, next_state, done = buffer.sample(batch_size=batch_size)

        state = torch.FloatTensor(state).to(device=self.device)
        action = torch.FloatTensor(action).to(device=self.device)
        reward = torch.FloatTensor(reward).to(device=self.device).unsqueeze(1)  # 扩展维度，行变为列
        next_state= torch.FloatTensor(next_state).to(device=self.device)
        done = torch.FloatTensor(done).to(device=self.device).unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = - self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.target_critic, self.critic, self.tau)
            soft_update(self.target_actor, self.actor, self.tau)

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