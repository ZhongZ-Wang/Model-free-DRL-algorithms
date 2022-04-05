import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from network import QnetDueling
import os


def hard_update(Q_target, Q):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data.copy_(param.data)


class PER(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(PER, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.Qnet = QnetDueling(state_dim, action_dim, args.hidden_size).to(device=self.device)
        self.target_Qnet = QnetDueling(state_dim, action_dim, args.hidden_size).to(device=self.device)
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
        q_value = self.Qnet(state)
        action = q_value.argmax(dim=1, keepdim=True) if np.random.rand() > self.epsilon else torch.randint(self.action_dim, size=(state.shape[0],1))
        return action.detach().cpu().numpy()[0].item()

    def update_parameter(self, buffer, batch_size, updates):
        idx_batch, batch_data, IS_weight_batch = buffer.sample(batch_size)
        state_batch = torch.FloatTensor(batch_data[:, :4]).to(device=self.device)
        action_batch = torch.FloatTensor(batch_data[:, 4:5]).to(device=self.device)
        reward_batch = torch.FloatTensor(batch_data[:, 5:6]).to(device=self.device) # 扩展维度，行变为列
        next_batch = torch.FloatTensor(batch_data[:, 6:10]).to(device=self.device)
        done_batch = torch.FloatTensor(batch_data[:, 10:11]).to(device=self.device)
        IS_weight_batch = torch.FloatTensor(IS_weight_batch).to(device=self.device)

        with torch.no_grad():
            next_action = self.Qnet(next_batch).argmax(dim=1, keepdim=True)
            next_q = self.target_Qnet(next_batch).gather(1, next_action.long())
            q_label= reward_batch + (1 - done_batch) * self.gamma * next_q

        q_value = self.Qnet(state_batch).gather(1, action_batch.long())
        td_errors = (q_label - q_value).abs().detach().cpu()

        loss = (F.mse_loss(q_value, q_label) * IS_weight_batch).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        buffer.update_priority(idx_batch, td_errors)

        self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)

        if updates % self.target_update_freq == 0:
            hard_update(self.target_Qnet, self.Qnet)

        return loss.item(), q_value.mean().item(), self.epsilon

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


