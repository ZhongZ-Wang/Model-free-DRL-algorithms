import numpy as np
import torch
from D3QN import D3QN
import gym
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import Replay_buffer
import argparse, datetime


def eval_policy(policy, env_name, seed, eval_episodes, episode_num):
    eval_env = gym.make(env_name)
    eval_env.seed(seed+10)
    avg_reward = 0
    for _ in range(eval_episodes):
        eval_state = eval_env.reset()
        eval_done = False
        while not eval_done:
            eval_action = policy.select_action(eval_state)
            eval_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
            avg_reward += eval_reward
    avg_reward /= eval_episodes

    writer.add_scalar('Test_avg_reward', avg_reward, episode_num + 1)
    print('----------')
    print(f'Evaluation over {eval_episodes} episodes: {avg_reward}')
    print('----------')
    return avg_reward


parser = argparse.ArgumentParser(description='Gym_Play')
parser.add_argument('--env_name', default='CartPole-v0',
                    help='CartPole-v1, MountainCar-v0, Acrobot-v1, LunarLander-v2, Pendulm-v0, MountainCarCountinous-v0')
parser.add_argument('--eval_episodes', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_epsilon', type=float, default=1)
parser.add_argument('--min_epsilon', type=int, default=0.01)
parser.add_argument('--epsilon_decay', type=float, default=1/2000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--target_update_freq', type=int, default=200)
parser.add_argument('--eval_freq', type=int, default=500)
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--replay_size', type=int, default=1000)
parser.add_argument('--start-steps', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=20000)
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()


env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = D3QN(env.observation_space.shape[0], env.action_space.n, args)

writer = SummaryWriter('../../runs/{}_D3QN_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), args.env_name))

memory = Replay_buffer(args.replay_size)

done = False
state = env.reset()
episode_timesteps = 0
episode_num = 0
episode_reward = 0
updates = 0

for t in range(args.max_timesteps):
    episode_timesteps += 1

    if  t < args.start_steps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)

    next_state, reward, done, _ = env.step(action)
    donebool = float(done) if episode_timesteps < env._max_episode_steps else 0
    memory.push(state, action, reward, next_state, donebool)
    state = next_state
    episode_reward += reward

    if len(memory) > args.batch_size:
        loss, q_value, epsilon = agent.update_parameter(memory, args.batch_size, updates)

        writer.add_scalar('Visualization/Train loss', loss, updates)
        writer.add_scalar('Visualization/Q_value', q_value, updates)
        writer.add_scalar('Visualization/Epsilon', epsilon, updates)
        updates += 1

    if done:
        print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        writer.add_scalar('visualization/Episode reward', episode_reward, t + 1)
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if (t + 1) % args.eval_freq == 0:
        score = eval_policy(agent, args.env_name, args.seed, args.eval_episodes, t)

    if (t + 1) % args.save_freq == 0:
        agent.save_model(args.env_name, path=None)

env.close()