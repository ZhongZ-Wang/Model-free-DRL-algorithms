import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from network import BetaActor, GaussianActor_mu, GaussianActor_musigma, Critic
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os, math


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(PPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = args.hidden_size
        self.dist = args.dist
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        if self.dist == 'Beta':
            self.actor = BetaActor(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)
        elif self.dist == 'GS_M':
            self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)
        else:
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.hidden_dim).to(device=self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic = Critic(self.state_dim, self.hidden_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

        self.eps_clip = args.eps_clip
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.k_epochs = args.k_epochs
        self.max_grad_norm = args.max_grad_norm
        self.l2_reg = args.l2_reg
        self.entropy_coef = args.entropy_coef
        self.entropy_coef_decay = args.entropy_coef_decay

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.data = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            dist = self.actor.get_dist(state)
            a = dist.sample()
            a = torch.clamp(a, 0, 1)
            a_logprob = dist.log_prob(a)
            return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def evaluate(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if self.dist == 'Beta':
                a = self.actor.mean(state)
            if self.dist == 'GS_MS':
                a, b = self.actor(state)
            if self.dist == 'GS_M':
                a = self.actor(state)
            return a.cpu().numpy().flatten()

    def update(self, memory):
        self.entropy_coef *= self.entropy_coef_decay
        s, a, a_logprob, r, s_n, dw, d = memory.numpy_to_tensor()
        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_n)
            deltas = r + self.gamma * (1 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(d.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            td_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        for _ in range(self.k_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])  # 包含了负值？
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv[index]
                loss1 = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor_optimizer.zero_grad()
                loss1.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                v_s = self.critic(s[index])
                loss2 = F.mse_loss(td_target[index], v_s)
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        loss2 += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                loss2.backward()
                self.critic_optimizer.step()

            return loss1.mean().item(), loss2.item()

    # def make_batch(self):
    #     s_list, a_list, a_logprob_list, r_list, s_n_list, d_list, dw_list = [], [], [], [], [], [], []
    #     for transition in self.data:
    #         s, a, a_logprob, r, s_n, d, dw = transition
    #         s_list.append(s)
    #         a_list.append(a)
    #         a_logprob_list.append(a_logprob)
    #         r_list.append(r)
    #         s_n_list.append(s_n)
    #         d_list.append([d])
    #         dw_list.append([dw])
    #
    #     # if not self.env_with_Dead:
    #     dw_list = (np.array(dw_list)*False).tolist()
    #
    #     self.data = []  # 清空数据
    #
    #     with torch.no_grad():
    #         s, a, a_logprob, r, s_n, d, dw = \
    #             torch.tensor(s_list, dtype=torch.float).to(device=self.device), \
    #             torch.tensor(a_list, dtype=torch.float).to(device=self.device), \
    #             torch.tensor(a_logprob_list, dtype=torch.float).to(device=self.device), \
    #             torch.tensor(r_list, dtype=torch.float).to(device=self.device), \
    #             torch.tensor(s_n_list, dtype=torch.float).to(device=self.device), \
    #             torch.tensor(d_list, dtype=torch.float).to(device=self.device), \
    #             torch.tensor(dw_list, dtype=torch.float).to(device=self.device)
    #     return s, a, a_logprob, r, s_n, d, dw
    #
    # def put_data(self, transition):
    #     self.data.append(transition)

    # def save_model(self, env_name, path=None):
    #     if not os.path.exists('../checkpoints/'):
    #         os.makedirs('../checkpoints/')
    #     if path is None:
    #         path = '../checkpoints/DDPG_checkpoint_{}'.format(env_name)
    #     torch.save({'actor_state_dict': self.actor.state_dict(),
    #                 'target_actor_state_dict': self.target_actor.state_dict(),
    #                 'critic_state_dict': self.critic.state_dict(),
    #                 'target_critic_state_dict': self.target_critic.state_dict(),
    #                 'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
    #                 'critic_optimizer_state_dict': self.critic_optimizer.state_dict()}, path)
    #
    # def load_model(self, path, evaluate=False):
    #     if path is not None:
    #         checkpoint = torch.load(path)
    #         self.Qnet.load_state_dict(checkpoint['Qnet_state_dict'])
    #         self.target_Qnet.load_state_dict(checkpoint['target_state_dict'])
    #         self.optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
    #     if evaluate:
    #         self.Qnet.eval()
    #         self.target_Qnet.eval()
    #     else:
    #         self.Qnet.train()
    #         self.target_Qnet.train()
