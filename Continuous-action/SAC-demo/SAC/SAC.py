import torch.nn as nn
import torch
import torch.nn.functional as F
from network import Actor, Critic
import copy


def hard_update(Q_target, Q):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data.copy_(param.data)


def soft_update(Q_target, Q, tau):
    for target_param, param in zip(Q_target.parameters(), Q.parameters()):
        target_param.data = (1 - tau) * target_param.data + tau * param.data


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super(SAC, self).__init__()
        self.action_dim = action_dim
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        self.target_entropy = -torch.prod(torch.tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr_alpha)

        self.actor = Actor(state_dim, action_dim, args.hidden_size).to(device=self.device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(state_dim, action_dim, args.hidden_size).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_freq = args.target_update_freq

    def select_action(self, state, is_test=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if is_test == False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameter(self, buffer, batch_size, updates):
        state, action, reward, next_state, done = buffer.sample(batch_size=batch_size)

        state = torch.FloatTensor(state).to(device=self.device)
        action = torch.FloatTensor(action).to(device=self.device)
        reward = torch.FloatTensor(reward).to(device=self.device).unsqueeze(1)  # 扩展维度，行变为列
        next_state = torch.FloatTensor(next_state).to(device=self.device)
        done = torch.FloatTensor(done).to(device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_logprob, _ = self.actor.sample(next_state)
            q1_next_target, q2_next_target = self.target_critic(next_state, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_logprob
            next_q = reward + (1 - done) * self.gamma * min_q_next_target

        q1, q2 = self.critic(state, action)
        q1_loss = F.mse_loss(q1, next_q)
        q2_loss = F.mse_loss(q2, next_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a, log_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()

        if updates % self.target_update_freq == 0:
            soft_update(self.target_critic, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

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