import numpy as np
import random
import os, pickle


class Replay_buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # 例子： array = [[1, 4], [2, 5], [3, 6]], map(list, zips(*array))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = 'checkpoints/DQN_buffer_{}'.format(env_name)

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)  # 序列化对象，并将结果数据流写入到文件对象中。

    def load_buffer(self, save_path):
        with open(save_path, 'rb') as f:
            self.buffer = pickle.load(f)  # 反序列化对象。将文件中的数据解析为一个Python对象
            self.position = len(self.buffer) % self.capacity