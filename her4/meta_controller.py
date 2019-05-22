import numpy as np
from collections import deque
import math


def arr_to_one_hot(arr, ncat):
    dtype = arr.dtype
    n = len(arr)
    _arr = np.zeros([n, ncat], dtype=dtype)
    index = (np.arange(n), arr.astype(int))
    _arr[index] = 1
    return _arr


def one_hot_to_arr(arr):
    index = np.where(arr)[1]
    _arr = np.array(index, dtype=arr.dtype)
    return _arr

def safe_mean(x):
    if len(x) == 0:
        return 0.
    else:
        return np.mean(x)


class MetaController:
    def __init__(self, maze_shape, goal_shape, goal_dtype):
        self.maze_shape = maze_shape
        self.goal_shape = goal_shape
        self.goal_dtype = goal_dtype
        self.maze_size = np.prod(maze_shape)
        self.ncat = goal_shape[1]
        self.n = goal_shape[0]
        self.p = np.ones(maze_shape, dtype=np.float32)
        self.t = np.zeros(maze_shape, dtype=deque)
        for i in range(maze_shape[0]):
            for j in range(maze_shape[1]):
                self.t[i][j] = deque(maxlen=5)
        self.desired_pos = arr_to_one_hot(np.array(maze_shape)-1, ncat=self.ncat)
        self.version = 2
        if self.version == 1:
            self.p[:] = np.sum(self.maze_shape)
        else:
            self.p = np.arange(0, self.maze_size).reshape(self.maze_shape)
            self.p = self.p / np.mean(self.p)
            self.p = self.p.astype(np.float32)

        self.possible_g = [arr_to_one_hot(np.array([0, 4]), self.ncat),
                           arr_to_one_hot(np.array([3, 8]), self.ncat),
                           arr_to_one_hot(np.array([3, 3]), self.ncat),
                           arr_to_one_hot(np.array([4, 0]), self.ncat),
                           arr_to_one_hot(np.array([9, 4]), self.ncat),
                           arr_to_one_hot(np.array([7, 8]), self.ncat),
                           arr_to_one_hot(np.array([9, 9]), self.ncat)]
        self.pointer = 0
        self.achieved_cnt = deque(maxlen=10)

        self.current_goal = self.possible_g[self.pointer]
        self.next_goal = self.possible_g[self.pointer + 1]

    def step_goal(self):
        if self.version == 1:
            loc = np.where(self.p == self.p.max())
            index = np.random.randint(len(loc[0]))
            row = loc[0][index]
            col = loc[1][index]
            goal = np.array([row, col])
        else:
            p = np.exp(-self.p).flatten()
            p = p / p.sum()
            index = np.random.choice(len(p), p=p)
            goal = np.unravel_index(index, self.maze_shape)

        # goal = np.array([4, 3])
        goal = np.array(goal)
        goal = arr_to_one_hot(goal, ncat=self.ncat)

        if safe_mean(self.achieved_cnt) > 0.7:
            self.pointer += 1
            self.pointer = min(self.pointer, len(self.possible_g)-1)
            self.achieved_cnt = deque(maxlen=10)
        self.current_goal = goal = self.possible_g[self.pointer]
        return goal

    def update(self, goal, final, t, alpha, beta=1.0):
        g = one_hot_to_arr(goal).astype(int)
        f = one_hot_to_arr(final).astype(int)
        d = one_hot_to_arr(self.desired_pos).astype(int)
        self.p *= beta
        if self.version == 1:
            bonus = np.sum(f)
        else:
            dist = np.sum(np.abs(g - d)) / d.sum()
            error = np.sum(np.abs(g - f)) / d.sum()
            time = t
            bonus = dist + error + time
            # bonus = np.clip(bonus, 0.001, 2.0)
        self.p[g[0], g[1]] = self.p[g[0], g[1]] * (1-alpha) + alpha * bonus

        self.achieved_cnt.append(np.array_equal(g, f))

    def sample_goal(self):
        goal = np.zeros(self.goal_shape, dtype=self.goal_dtype)
        index = np.random.choice(self.ncat, size=self.n)
        index = (np.arange(self.n), index)
        goal[index] = 1
        return goal
