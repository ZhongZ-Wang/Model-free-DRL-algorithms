import numpy as np
import torch
from PPO import PPO
import gym
from torch.utils.tensorboard import SummaryWriter
from utils import Reward_adapter, Action_adapter
import argparse, datetime
from replay_buffer import Memory


def eval_policy(policy, env_name, seed, eval_episodes, t):
    eval_env = gym.make(env_name)
    eval_env.seed(seed+10)
    avg_reward = 0
    for _ in range(eval_episodes):
        eval_state = eval_env.reset()
        eval_done = False
        while not eval_done:
            eval_action = policy.evaluate(eval_state)
            eval_act = Action_adapter(eval_action, float(env.action_space.high[0]))
            eval_state, eval_reward, eval_done, _ = eval_env.step(eval_act)
            avg_reward += eval_reward
    avg_reward /= eval_episodes

    writer.add_scalar('Test_avg_reward', avg_reward, t + 1)
    print('----------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward}')
    print('----------')
    return avg_reward


parser = argparse.ArgumentParser(description='Gym_Play')
# '''
# Parameters for Pendulum-v0
parser.add_argument('--env_name', default='Hopper-v2', help='LunarLanderContinuous-v2, Pendulum-v0, MountainCarCountinous-v0')
parser.add_argument('--eval_episodes', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lamda', type=float, default=0.95)
parser.add_argument('--k_epochs', type=float, default=10)
parser.add_argument('--lr_actor', type=float, default=0.0003)
parser.add_argument('--lr_critic', type=float, default=0.0003)
parser.add_argument('--entropy_coef', type=float, default=1e-3)
parser.add_argument('--entropy_coef_decay', type=float, default=0.99)
parser.add_argument('--max_grad_norm', type=float, default=40)
parser.add_argument('--l2_reg', type=float, default=1e-3)
parser.add_argument('--eps_clip', type=float, default=0.2)
parser.add_argument('--dist', default='GS_M', help='Beta, GS_M, GS_MS')
parser.add_argument('--mini_batch_size', type=float, default=64)
parser.add_argument('--batch_size', type=float, default=2048)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--eval_freq', type=int, default=5000)
parser.add_argument('--save_freq', type=int, default=10000)
parser.add_argument('--max_timesteps', type=int, default=100000)
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()


env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = PPO(env.observation_space.shape[0], env.action_space.shape[0], args)

memory = Memory(env.observation_space.shape[0], env.action_space.shape[0], args.batch_size)

writer = SummaryWriter('../../runs_{}/{}_PPO_{}'.format(args.env_name, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), args.env_name))

evaluations = [eval_policy(agent, args.env_name, args.seed, args.eval_episodes, 0)]

done = False
state = env.reset()
episode_timesteps = 0
episode_num = 0
episode_reward = 0
for t in range(args.max_timesteps):
    episode_timesteps += 1
    action, action_logprob = agent.select_action(state)
    act = Action_adapter(action, float(env.action_space.high[0]))
    next_state, reward, done, _ = env.step(act)
    reward = Reward_adapter(reward)

    if done and episode_timesteps != env._max_episode_steps:
        dw = True
    else:
        dw = False

    memory.store(state, action, action_logprob, reward, next_state, dw, done)
    state = next_state
    episode_reward += reward

    if done:
        print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        writer.add_scalar('visualization/Episode reward', episode_reward, t + 1)
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if memory.count == args.batch_size:
        actor_loss, critic_loss = agent.update(memory)
        writer.add_scalar('Visualization/critic loss', critic_loss, t + 1)
        writer.add_scalar('Visualization/actor loss', actor_loss, t + 1)
        memory.count = 0

    if (t + 1) % args.eval_freq == 0:
        score = eval_policy(agent, args.env_name, args.seed, args.eval_episodes, t + 1)
        evaluations.append(score)

np.save(f'../result/test_score', evaluations)

env.close()