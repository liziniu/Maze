import numpy as np
from baselines.common.runners import AbstractEnvRunner
from gym import spaces
import pickle
from baselines import logger
from common.util import DataRecorder
import os
from copy import deepcopy
from her3.meta_controller import arr_to_one_hot


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, save_interval):
        super().__init__(env=env, model=model, nsteps=nsteps)
        assert isinstance(env.action_space, spaces.Discrete), 'This ACER implementation works only with discrete action spaces!'

        self.nact = env.action_space.n
        nenv = self.nenv
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1),) + env.observation_space.shape

        # self.obs = env.reset()
        self.obs_dtype = env.observation_space.dtype
        self.obs_shape = env.observation_space.shape
        self.ac_dtype = env.action_space.dtype

        self.recoder = DataRecorder(os.path.join(logger.get_dir(), "runner_data"))
        self.save_interval = save_interval

        self.size = [int(x) for x in self.env.spec.id.split("-")[2].split("x")]
        self.desired_pos = np.asarray(self.size) - 1
        self.ncat = self.size[0]
        logger.info("-"*50)
        logger.info("-"*15, "desired_pos:", self.desired_pos, "-"*15)
        logger.info("-"*50)

        self.goals, self.goal_infos = self.get_goal(self.nenv)
        self.episode_step = np.zeros(self.nenv, dtype=np.int32)
        self.episode = np.zeros(self.nenv, dtype=np.int32)

    def run(self, debug=False):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        mb_obs, mb_next_obs, mb_actions, mb_mus, mb_dones, mb_rewards, mb_goals = [], [], [], [], [], [], [],
        episode_info = {}
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            if debug:
                self.env.render()
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            mb_goals.append(np.copy(self.goals))

            # this for maze env because we edify step fn.
            inputs = []
            for i in range(self.nenv):
                inputs.append({
                    'action': actions[i],
                    'done_block': False,
                })
            obs, rewards, dones, infos = self.env.step(inputs)
            self.episode_step += 1
            for env_idx in range(self.nenv):
                if dones[env_idx]:
                    self.episode_step[env_idx] = 0
                    self.episode[env_idx] += 1
                    episode_info["episode"] = infos[env_idx]["episode"]
                    assert np.array_equal(obs[env_idx], arr_to_one_hot(np.array([0., 0.]), self.ncat)), "next_obs:{}".format(obs[env_idx])
                    next_obs_i = infos[env_idx].get("next_obs", None)
                    assert next_obs_i is not None
                    next_obs = obs.copy()
                    next_obs[env_idx] = next_obs_i
                    mb_next_obs.append(next_obs)
                else:
                    mb_next_obs.append(obs)
                if rewards[env_idx] == 1.0:
                    assert np.array_equal(obs[env_idx], arr_to_one_hot(np.array([0., 0.]), self.ncat))
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(np.copy(self.dones))

        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_goals = np.asarray(mb_goals, dtype=self.goals.dtype).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        results = dict(
            obs=mb_obs,
            next_obs=mb_next_obs,
            actions=mb_actions,
            rewards=mb_rewards,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            goal_obs=mb_goals,
            episode_info=episode_info,
        )
        return results

    def get_goal(self, nb_goal):
        return np.array([arr_to_one_hot(self.desired_pos, self.ncat) for _ in range(nb_goal)]), [{} for _ in range(nb_goal)]

    def initialize(self):
        pass

    @staticmethod
    def check_goal_reached_v2(obs_info, goal_info):
        pass

