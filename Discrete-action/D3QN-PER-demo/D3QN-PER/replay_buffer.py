import numpy as np
import random
import os, pickle
from SumTree import sumtree


class Per_replay_buffer:
    def __init__(self, capacity, alpha, beta, beta_incre, prio_max, prio_e):
        self.sumtree = sumtree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_incre = beta_incre
        self.prio_max = prio_max
        self.prio_e = prio_e
        self.length = 0

    def push(self, state, action, reward, next_state, done):
        self.length += 1
        data = np.hstack((state, action, reward, next_state, done))  # 把所有值放入一行 （-1，11）
        max_p = np.max(self.sumtree.tree[-self.sumtree.capacity:])
        if max_p == 0:
            max_p = self.prio_max
        self.sumtree.add(max_p, data)

    def sample(self, batch_size):
        batch_idx, batch, IS_weights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, self.sumtree.data[0].size)), np.empty((batch_size,1))
        segment = self.sumtree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_incre])
        p_sum = self.sumtree.tree[0]
        min_prob = np.min(self.sumtree.tree[-self.sumtree.capacity:]) / p_sum
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.sumtree.get(s)
            prob = p / p_sum
            IS_weights[i, 0] = np.power(min_prob/prob, self.beta)
            batch_idx[i], batch[i, :] = idx, data

        return batch_idx, batch, IS_weights

    def update_priority(self, idxs, td_errors):
        td_errors += self.prio_e
        clipped_errors = np.minimum(td_errors, 1.0)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(idxs, ps):
            self.sumtree.update(ti, p)

    def size(self):
        return self.length

    def save_buffer(self, env_name, save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = 'checkpoints/PER_buffer_{}'.format(env_name)

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)  # 序列化对象，并将结果数据流写入到文件对象中。

    def load_buffer(self, save_path):
        with open(save_path, 'rb') as f:
            self.buffer = pickle.load(f)  # 反序列化对象。将文件中的数据解析为一个Python对象
            self.position = len(self.buffer) % self.sumtree.capacity