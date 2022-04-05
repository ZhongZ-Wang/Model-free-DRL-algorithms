import numpy as np
import torch
from DDQN import DDQN
import gym
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import Replay_buffer
import argparse, datetime


parser = argparse.ArgumentParser(description='Gym_Play')
parser.add_argument('--env_name', default='CartPole-v0',
                    help='CartPole-v1, MountainCar-v0, Acrobot-v1, Pendulm-v0, MountainCarCountinous-v0')
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_epsilon', type=float, default=1)
parser.add_argument('--min_epsilon', type=int, default=0.1)
parser.add_argument('--epsilon_decay', type=float, default=1/2000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=5)
parser.add_argument('--target_update_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--replay_size', type=int, default=3000)
parser.add_argument('--start-steps', type=int, default=200)
parser.add_argument('--max_episodes', type=int, default=300)
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()


env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = DDQN(env.observation_space.shape[0], env.action_space.n, args)

writer = SummaryWriter('../../runs/{}_DQN_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), args.env_name))

memory = Replay_buffer(args.replay_size)

total_numsteps = 0
updates = 0
for episode in range(args.max_episodes):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    # env.render()
    while not done:
        if  total_numsteps < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                loss, q_value, epsilon = agent.update_parameter(memory, args.batch_size, updates)

                writer.add_scalar('Visualization/Train loss', loss, updates)
                writer.add_scalar('Visualization/Q_value', q_value, updates)
                writer.add_scalar('Visualization/Epsilon', epsilon, updates)

                updates += 1

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask)
        state = next_state

    writer.add_scalar('visualization/Episode reward', episode_reward, episode)
    print('Episode: {}, total numsteps: {}, total reward: {}'.format(episode, total_numsteps, round(episode_reward, 2)))

    if episode % 10 == 0 and args.eval == True:
        avg_reward = 0
        test_episodes = 10
        for _ in range(test_episodes):
            state = env.reset()
            ep_reward = 0
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
            avg_reward += ep_reward
        avg_reward /= test_episodes

        writer.add_scalar('Test_avg_reward', avg_reward, episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(test_episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    if episode % args.save_freq:
        agent.save_model(args.env_name, path=None)

env.close()