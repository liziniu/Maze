import gym
import numpy as np
import pickle
import json
from gym.utils import seeding
from gym import spaces
from gym.envs.registration import register


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    DIRECTION = {
        0: "N",
        1: "S",
        2: "W",
        3: "E"
    }

    RESIDUAL = {
        "N": np.array([0, -1]),
        "S": np.array([0, +1]),
        "W": np.array([-1, 0]),
        "E": np.array([+1, 0])
    }

    def __init__(self, maze_size, wall_file=None):
        if isinstance(maze_size, int):
            maze_size = (maze_size, maze_size)
        else:
            assert isinstance(maze_size, tuple) or isinstance(maze_size, list)
            assert len(maze_size) == 2
        self.maze_size = maze_size
        self.cell = np.zeros(shape=self.maze_size, dtype=object)
        self.wall = np.zeros(shape=self.maze_size, dtype=dict)
        self.W, self.H = self.maze_size
        for i in range(self.W):
            for j in range(self.H):
                self.wall[i][j] = dict(N=False, S=False, W=False, E=False)
        self._init_wall()
        self.state = np.array([0., 0.])
        if wall_file is not None:
            f = open(wall_file, 'rb')
            walls = pickle.load(f)
            assert isinstance(walls, list)
            for wall in walls:
                index, direction = wall['index'], wall['direction']
                self.wall[index][direction] = True

        self.target = np.array(self.maze_size) - 1

        self.action_space = spaces.Discrete(2*len(self.maze_size))
        low = np.zeros(len(self.maze_size), dtype=int)
        high = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high)

    def _init_wall(self):
        for j in range(self.H):
            self.wall[0][j]["W"] = True
        for j in range(self.H):
            self.wall[self.W-1][j]["E"] = True
        for i in range(self.W):
            self.wall[i][0]["N"] = True
        for i in range(self.W):
            self.wall[i][self.H-1]["S"] = True

    def step(self, action):
        action = int(action)
        assert action in [0, 1, 2, 3]
        d = self.DIRECTION[action]
        res = self.RESIDUAL[d]
        i, j = int(self.state[0]), int(self.state[1])
        if not self.wall[i][j][d]:
            self.state += res
        else:
            self.state = self.state
        assert 0 <= self.state[0] <= self.W-1 and 0 <= self.state[1] <= self.H-1
        if np.array_equal(self.state, self.target):
            reward = 1.0
            done = True
        else:
            reward = -0.1 / np.prod(self.maze_size)
            done = False
        info = {}
        return self.state.copy(), reward, done, info

    def reset(self):
        self.state = np.array([0., 0.])
        return self.state.copy()

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


register(
    id='simple-maze-5x5-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=250,
    kwargs={'maze_size': (5, 5)}
)

register(
    id='simple-maze-6x6-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=360,
    # timestep_limit=50,
    kwargs={'maze_size': (6, 6)}
)

register(
    id='simple-maze-7x7-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=490,
    # timestep_limit=50,
    kwargs={'maze_size': (7, 7)}
)

register(
    id='simple-maze-8x8-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=640,
    # timestep_limit=50,
    kwargs={'maze_size': (8, 8)}
)

register(
    id='simple-maze-9x9-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=810,
    # timestep_limit=50,
    kwargs={'maze_size': (9, 9)}
)

register(
    id='simple-maze-10x10-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=50,
    # timestep_limit=50,
    kwargs={'maze_size': (10, 10)}
)

register(
    id='simple-maze-20x20-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=400,
    # timestep_limit=50,
    kwargs={'maze_size': (20, 20)}
)

register(
    id='median-maze-20x20-v0',
    entry_point='common.maze:MazeEnv',
    timestep_limit=400,
    # timestep_limit=50,
    kwargs={'maze_size': (20, 20), 'wall_file': 'common/20x20_median.pkl'}
)

if __name__ == "__main__":
    #    0  1   2  3  4
    #    __ __ __ __ __
    # 0 |__    __    __|
    # 1 |              |
    # 2 |  |  |  |  |  |
    # 3 |   __    __   |
    # 4 |__ __ __ __ __|

    offset = (15, 15)
    wall_content = []
    # 0th row
    wall_content.append(dict(index=(0, 0), direction='S'))
    wall_content.append(dict(index=(0, 1), direction='N'))

    wall_content.append(dict(index=(2, 0), direction='S'))
    wall_content.append(dict(index=(2, 1), direction='N'))

    wall_content.append(dict(index=(4, 0), direction='S'))
    wall_content.append(dict(index=(4, 1), direction='N'))

    # 2th row
    wall_content.append(dict(index=(0, 2), direction='E'))
    wall_content.append(dict(index=(1, 2), direction='W'))

    wall_content.append(dict(index=(1, 2), direction='E'))
    wall_content.append(dict(index=(2, 2), direction='W'))

    wall_content.append(dict(index=(2, 2), direction='E'))
    wall_content.append(dict(index=(3, 2), direction='W'))

    wall_content.append(dict(index=(3, 2), direction='E'))
    wall_content.append(dict(index=(4, 2), direction='W'))

    # 3th row
    wall_content.append(dict(index=(1, 3), direction='S'))
    wall_content.append(dict(index=(1, 4), direction='N'))

    wall_content.append(dict(index=(3, 3), direction='S'))
    wall_content.append(dict(index=(4, 3), direction='N'))

    for wall in wall_content:
        wall['index'] = (wall['index'][0] + offset[0], wall['index'][1] + offset[1])

    wall_f = open('20x20_median.json', 'w')
    json.dump(wall_content.copy(), wall_f, indent=4)
    wall_f.close()
    wall_f = open('20x20_median.pkl', 'wb')
    pickle.dump(wall_content.copy(), wall_f)
    wall_f.close()

    env = MazeEnv(maze_size=20, wall_file='20x20_median.pkl')
    obs = env.reset()
    while True:
        action = input()
        while action not in ['0', '1', '2', '3']:
            action = input()
        action = int(action)
        obs_new, reward, done, info = env.step(action)
        print('-----------------------------')
        print('obs:', obs)
        print('action:', env.DIRECTION[action])
        print('next_obs:', obs_new)
        print('-----------------------------')