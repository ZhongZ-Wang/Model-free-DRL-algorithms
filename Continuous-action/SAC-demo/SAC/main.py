import numpy as np
import torch
from SAC import SAC
import gym
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import Replay_buffer
import argparse, datetime


def eval_policy(policy, env_name, seed, eval_episodes, t):
    eval_env = gym.make(env_name)
    eval_env.seed(seed+10)
    avg_reward = 0
    for _ in range(eval_episodes):
        eval_state = eval_env.reset()
        eval_done = False
        while not eval_done:
            eval_action = policy.select_action(eval_state, is_test=True)
            eval_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
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
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--lr_actor', type=float, default=0.0003)
parser.add_argument('--lr_critic', type=float, default=0.0003)
parser.add_argument('--lr_alpha', type=float, default=0.0003)
parser.add_argument('--update_per_step', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--target_update_freq', type=int, default=1)
parser.add_argument('--eval_freq', type=int, default=5000)
parser.add_argument('--save_freq', type=int, default=10000)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--start-steps', type=int, default=5000)
parser.add_argument('--max_timesteps', type=int, default=35000)
parser.add_argument('--cuda', type=bool, default=False)
args = parser.parse_args()


env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = SAC(env.observation_space.shape[0], env.action_space.shape[0], args)

writer = SummaryWriter('../../runs_{}/{}_SAC_{}'.format(args.env_name, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), args.env_name))

memory = Replay_buffer(args.replay_size)

evaluations = [eval_policy(agent, args.env_name, args.seed, args.eval_episodes, 0)]

done = False
state = env.reset()
episode_timesteps = 0
episode_num = 0
episode_reward = 0
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
        for i in range(args.update_per_step):
            critic_loss, actor_loss, alpha_loss, alpha = agent.update_parameter(memory, args.batch_size, t)
            writer.add_scalar('Visualization/critic loss', critic_loss, t)
            writer.add_scalar('Visualization/actor loss', actor_loss, t)
            writer.add_scalar('Visualization/alpha loss', alpha_loss, t)
            writer.add_scalar('Visualization/alpha', alpha, t)

    if done:
        print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        writer.add_scalar('visualization/Episode reward', episode_reward, t + 1)
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    if (t + 1) % args.eval_freq == 0:
        score = eval_policy(agent, args.env_name, args.seed, args.eval_episodes, t)
        evaluations.append(score)

    # if (t + 1) % args.save_freq == 0:
    #     agent.save_model(args.env_name, path=None)

np.save(f'../result/test_score', evaluations)

env.close()