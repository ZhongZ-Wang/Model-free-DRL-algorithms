import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from network import Actor, Critic


def soft_update(Q_target, Q, tau):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data = (1 - tau) * target_param.data + tau * param.data


class REDQ(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(REDQ, self).__init__()
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.gamma = args.gamma
        self.tau = args.tau
        self.N = args.N  # Number of Q-Network Ensemble. Defaults to 2.
        self.M = args.M  # Number of the subset of the Critic for update calculation. Defaults to 2.
        self.G = args.G  # Critic Updates per step, UTD-raio. Defaults to 1.
        self.target_q_mode = args.target_q_mode
        # self.policy_update_decay = args.policy_update_decay

        self.actor = Actor(state_dim, action_dim, args.hidden_size).to(device=self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critics = []
        self.target_critics = []
        self.critic_optimizers = []
        for i in range(self.N):
            critic = Critic(state_dim, action_dim, args.hidden_size).to(device=self.device)
            self.critics.append(critic)
            target_critic = Critic(state_dim, action_dim, args.hidden_size).to(device=self.device)
            self.target_critics.append(target_critic)
            self.critic_optimizers.append(torch.optim.Adam(critic.parameters(), args.lr_critic))

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr_alpha)

    def select_action(self, state, is_test=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if is_test == False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameter(self, buffer, batch_size):
        for i_update in range(self.G):
            state, action, reward, next_state, done = buffer.sample(batch_size=batch_size)

            state = torch.FloatTensor(state).to(device=self.device)
            action = torch.FloatTensor(action).to(device=self.device)
            reward = torch.FloatTensor(reward).to(device=self.device).unsqueeze(1)  # 扩展维度，行变为列
            next_state = torch.FloatTensor(next_state).to(device=self.device)
            done = torch.FloatTensor(done).to(device=self.device).unsqueeze(1)

            idxs = np.random.choice(self.N, self.M, replace=False)
            with torch.no_grad():
                if self.target_q_mode == 'min':
                    next_action, next_log_prob, _ = self.actor.sample(next_state)
                    q_target_next_list = []
                    for idx in idxs:
                        q_target_next_idx = self.target_critics[idx](next_state, next_action)
                        q_target_next_list.append(q_target_next_idx)  # 每个列表正有256个元素
                    q_target_next_cat = torch.cat(q_target_next_list, 1)  # 数据形状是否存在问题？
                    q_target_next = torch.min(q_target_next_cat, dim=1, keepdim=True)[0] - self.alpha * next_log_prob
                    q_target = reward + self.gamma * (1 - done) * q_target_next

                if self.target_q_mode == 'avg':
                    next_action, next_log_prob, _ = self.actor.sample(next_state)
                    q_target_next_list = []
                    for idx in range(self.N):
                        q_target_next_idx = self.target_critics[idx](next_state, next_action)
                        q_target_next_list.append(q_target_next_idx)
                    q_target_next_cat = torch.cat(q_target_next_list, 1)
                    q_target_next = torch.mean(q_target_next_cat, dim=1, keepdim=True)[0] - self.alpha * next_log_prob
                    q_target = reward + self.gamma * (1 - done) * q_target_next

                if self.target_q_mode == 'rem':  # 随机集成
                    next_action, next_log_prob, _ = self.actor.sample(next_state)
                    q_target_next_list = []
                    for idx in range(self.N):
                        q_target_next_idx = self.target_critics[idx](next_state, next_action)
                        q_target_next_list.append(q_target_next_idx)
                    q_target_next_cat = torch.cat(q_target_next_list, 1)
                    rem_weight = torch.tensor(np.random.uniform(0, 1, q_target_next_cat.shape)).to(device=self.device)
                    normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.N)
                    rem_weight = rem_weight / normalize_sum
                    q_target_next = (q_target_next_cat * rem_weight).sum(dim=1).reshape(-1, 1) - self.alpha * next_log_prob
                    q_target = reward + self.gamma * (1 - done) * q_target_next

            q_pred_list = []
            for i in range(self.N):
                q_pred = self.critics[i](state, action)
                q_pred_list.append(q_pred)
            q_pred_cat = torch.cat(q_pred_list, 1)
            q_target = q_target.expand((-1, self.N)) if q_target.shape[1] == 1 else q_target

            critic_loss_all = F.mse_loss(q_pred_cat, q_target) * self.N
            for i in range(self.N):
                self.critic_optimizers[i].zero_grad()
            critic_loss_all.backward()

            # if (i_update + 1) % self.policy_update_decay == 0 or i_update == self.G - 1:
            if i_update == self.G - 1:
                action, log_prob, _ = self.actor.sample(state)
                q_list = []
                for idx in range(self.N):
                    self.critics[idx].requires_grad_(False)
                    q = self.critics[idx](state, action)
                    q_list.append(q)
                q_cat = torch.cat(q_list, 1)
                avg_q = torch.mean(q_cat, dim=1, keepdim=True)
                actor_loss = (self.alpha * log_prob - avg_q).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                for idx in range(self.N):
                    self.critics[idx].requires_grad_(True)

                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.cpu().exp().item()

            for i in range(self.N):
                self.critic_optimizers[i].step()

            # if (i_update + 1) % self.policy_update_decay == 0 or i_update == self.G - 1:
            if i_update == self.G - 1:
                self.actor_optimizer.step()

            for i in range(self.N):
                soft_update(self.target_critics[i], self.critics[i], self.tau)