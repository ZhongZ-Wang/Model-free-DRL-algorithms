import numpy as np
import random, torch
import os, pickle


class Memory():
    def __init__(self, state_dim, action_dim, batch_size):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, action_dim))
        self.r = np.zeros((batch_size, 1))
        self.s_n = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.d = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_n, dw, d):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_n[self.count] = s_n
        self.dw[self.count] = dw
        self.d[self.count] = d
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_n = torch.tensor(self.s_n, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        d = torch.tensor(self.d, dtype=torch.float)
        return s, a, a_logprob, r, s_n, dw, d

    # def save_buffer(self, env_name, save_path=None):
    #     if not os.path.exists('checkpoints/'):
    #         os.makedirs('checkpoints/')
    #
    #     if save_path is None:
    #         save_path = 'checkpoints/DDPG_buffer_{}'.format(env_name)
    #
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(self.buffer, f)  # 序列化对象，并将结果数据流写入到文件对象中。
    #
    # def load_buffer(self, save_path):
    #     with open(save_path, 'rb') as f:
    #         self.buffer = pickle.load(f)  # 反序列化对象。将文件中的数据解析为一个Python对象
    #         self.position = len(self.buffer) % self.capacity