import numpy as np


class sumtree:  # 碎片资料 强化学习代码实践（二）
    def __init__(self, capacity):
        self.write = 0  # 指出元素在最后一层的位置
        self.capacity = capacity  # 最深层的节点个数 == 经验样本池的容量
        self.tree = np.zeros(2 * capacity - 1)   #  整个数所有的节点个数优先值
        self.data = np.zeros(capacity, dtype=object)  # 存储样本池的元组数据  object 对象的使用

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1  # 元素在整个树中的位置
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:  # this method is faster than the recursive loop in the reference code
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.tree)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if s <= self.tree[l_idx]:
                    parent_idx = l_idx
                else:
                    s -= self.tree[l_idx]
                    parent_idx = r_idx

        dataIdx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[dataIdx]