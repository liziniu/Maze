import numpy as np
from collections import deque
import math


class MetaController:
    def __init__(self, maze_shape):
        self.maze_shape = maze_shape
        self.maze_size = np.prod(maze_shape)
        self.p = np.zeros(maze_shape, dtype=np.float32)
        self.t = np.zeros(maze_shape, dtype=deque)
        for i in range(maze_shape[0]):
            for j in range(maze_shape[1]):
                self.t[i][j] = deque(maxlen=5)
        self.desired_pos = np.array(maze_shape) - 1
        self.version = 2
        if self.version == 1:
            self.p[:] = np.sum(self.maze_shape)

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

        return np.array(goal)

    def update(self, goal, final,  alpha, beta=1.0):
        g = goal.astype(int)
        f = final.astype(int)
        self.p *= beta
        if self.version == 1:
            bonus = np.sum(f)
        else:
            bonus = np.sum(self.desired_pos - f)
        self.p[g[0], g[1]] = self.p[g[0], g[1]] * (1-alpha) + alpha * bonus

    def initial_goal(self):
        goal = []
        for sh in self.maze_shape:
            goal.append(np.random.randint(sh))
        return np.asarray(goal, dtype=np.int32)

    def initial_step(self):
        return np.random.randint(np.max(self.maze_shape), np.prod(self.maze_shape))