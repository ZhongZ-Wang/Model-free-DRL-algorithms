import torch.nn as nn
import torch, os
import numpy as np
from network import Qnet


def hard_update(Q_target, Q):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data.copy_(param.data)


class C51(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(C51, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.Qnet = Qnet(state_dim, action_dim, args.hidden_size).to(device=self.device)
        self.target_Qnet = Qnet(state_dim, action_dim, args.hidden_size).to(device=self.device)
        hard_update(self.target_Qnet, self.Qnet)
        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.epsilon = args.max_epsilon
        self.max_epsilon = args.max_epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.target_update_freq = args.target_update_freq

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist, q_value = self.Qnet(state)
        action = q_value.argmax(dim=1, keepdim=True) if np.random.rand() > self.epsilon else torch.randint(self.action_dim, size=(state.shape[0],1))
        return action.detach().cpu().numpy()[0].item()

    def update_parameter(self, buffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_batch, done_batch = buffer.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(device=self.device)
        action_batch = torch.FloatTensor(action_batch).to(device=self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).to(device=self.device).unsqueeze(1)  # 扩展维度，行变为列
        next_batch = torch.FloatTensor(next_batch).to(device=self.device)
        done_batch = torch.FloatTensor(done_batch).to(device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_dist, next_q = self.target_Qnet(next_batch)
            next_actions = torch.max(next_q, 1)[1] # 取出数值对应的索引，即为动作
            next_dist = next_dist[range(batch_size), next_actions]

            t_z = reward_batch + (1 - done_batch) * self.gamma * self.Qnet.support
            t_z = t_z.clamp(min=self.Qnet.vmin, max=self.Qnet.vmax)
            delta_z = (self.Qnet.vmax - self.Qnet.vmin) / (self.Qnet.n_atoms - 1)
            b = (t_z - self.Qnet.vmin) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * self.Qnet.n_atoms, batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(batch_size, self.Qnet.n_atoms)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        old_dist, q_value = self.Qnet(state_batch)
        log_p = torch.log(old_dist[range(batch_size), action_batch.long()])

        loss = - (proj_dist * log_p).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)

        if updates % self.target_update_freq == 0:
            hard_update(self.target_Qnet, self.Qnet)

        return loss.item(), self.epsilon

    def save_model(self, env_name, path=None):
        if not os.path.exists('../checkpoints/'):
            os.makedirs('../checkpoints/')
        if path is None:
            path = '../checkpoints/DQN_checkpoint_{}'.format(env_name)
        torch.save({'Qnet_state_dict': self.Qnet.state_dict(),
                    'target_Qnet_state_dict': self.target_Qnet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)

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