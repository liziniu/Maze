import numpy as np
from gym import spaces
from her3.meta_controller import arr_to_one_hot


class EvalRunner:
    def __init__(self, env, model):
        assert isinstance(env.action_space,
                          spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'
        self.obs = env.reset()
        self.env = env
        self.nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.model = model
        self.maze_shape = [int(x) for x in self.env.spec.id.split("-")[2].split("x")]
        self.desired_pos = arr_to_one_hot(np.asarray(self.maze_shape) - 1, ncat=self.maze_shape[0])
        self.goals = np.array([self.desired_pos for _ in range(self.nenv)])
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]

    def evaluate(self):
        self.obs = self.env.reset()
        self.goals = np.array([self.desired_pos for _ in range(self.nenv)])
        dones = [False for _ in range(self.nenv)]
        while not dones[0]:
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            # this will influence reset by vec_env.
            inputs = []
            for env_idx in range(self.nenv):
                inputs.append({
                    'action': actions[env_idx],
                    'done_block': False,
                })
            obs, rewards, dones, infos = self.env.step(inputs)
            self.obs = obs
            self.dones = dones
        # this is correct because default goal is same with desired_pos.
        episode_info = infos[0].get("episode")
        return episode_info
