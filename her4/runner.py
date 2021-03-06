import numpy as np
from baselines.common.runners import AbstractEnvRunner
from gym import spaces
from baselines import logger
from common.util import DataRecorder
import os
from her4.meta_controller import MetaController, one_hot_to_arr, arr_to_one_hot


def get_entropy(mu):
    return np.mean(-np.sum(mu * np.log(mu + 1e-6), axis=1))


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps, total_steps, save_interval, her):
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

        self.total_steps = total_steps

        self.maze_shape = [int(x) for x in self.env.spec.id.split("-")[2].split("x")]
        self.desired_pos = arr_to_one_hot(np.asarray(self.maze_shape) - 1, ncat=self.maze_shape[0])
        logger.info("-"*50)
        logger.info("-"*15, "desired_pos:", self.desired_pos, "-"*15)
        logger.info("-"*50)

        self.her = her

        assert self.nenv == 1
        self.controller = MetaController(self.maze_shape, env.observation_space.shape, env.observation_space.dtype)
        self.allowed_step = [np.prod(self.maze_shape)*10]
        self.allowed_step = [np.inf]
        self.goal_infos = [{}]
        self.goals = np.array([self.controller.sample_goal()])
        self.aux_goal = np.copy(self.goals[0])
        self.mem = ""

        self.episode_step = np.zeros(self.nenv, dtype=np.int32)
        self.episode = np.zeros(self.nenv, dtype=np.int32)
        self.aux_step = np.zeros(self.nenv, dtype=np.int32)
        self.aux_dones = np.empty(self.nenv, dtype=bool)
        self.max_episode_length = 3000
        self.aux_dones.fill(False)
        self.aux_entropy = 0.
        self.tar_entropy = 0.

    def run(self, acer_step=None, debug=False):
        # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        mb_obs, mb_next_obs, mb_actions, mb_mus, mb_dones, mb_masks, mb_rewards, mb_goals = [], [], [], [], [], [], [], []
        mb_aux = []
        episode_info = {}
        entropy = []
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step(self.obs, S=self.states, M=self.dones, goals=self.goals)
            entropy.append(get_entropy(mus))
            if debug:
                self.env.render()
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_masks.append(self.dones)
            mb_goals.append(np.copy(self.goals))
            mb_aux.append([True])
            inputs = []
            for env_idx in range(self.nenv):
                inputs.append({
                    'action': actions[env_idx],
                    'done_block': True if not self.aux_dones[env_idx] else False,
                })
            obs, _, dones, infos = self.env.step(inputs)
            self.episode_step += 1

            # next obs
            next_obs = obs.copy()
            for env_idx in range(self.nenv):
                if dones[env_idx]:
                    o = infos[env_idx].get("next_obs", None)
                    assert o is not None
                    next_obs[env_idx] = o
            mb_next_obs.append(next_obs)
            # rewards
            rewards = []
            for env_idx in range(self.nenv):
                if np.array_equal(next_obs[env_idx], self.goals[env_idx]):
                    r = 1.0
                else:
                    r = -0.1 / np.prod(self.maze_shape)
                rewards.append(r)

            for env_idx in range(self.nenv):
                real_done = dones[env_idx]
                if not self.aux_dones[env_idx]:
                    if np.array_equal(next_obs[env_idx], self.goals[env_idx]):
                        dones[env_idx] = True
                        self.controller.update(self.goals[env_idx], next_obs[env_idx],
                                               self.episode_step[env_idx]/self.max_episode_length,
                                               acer_step / self.total_steps)
                        self.aux_dones[env_idx] = True
                        self.aux_step[env_idx] = self.episode_step[env_idx]
                        aux_goal = one_hot_to_arr(self.aux_goal)
                        episode_info["aux_info"] = dict(aux_x=aux_goal[0],
                                                        aux_y=aux_goal[1],
                                                        succ=True)
                        self.mem = "aux_succ:True, real_step:{}".format(self.aux_step[env_idx])
                        self.goals[env_idx] = self.desired_pos
                        self.aux_entropy = np.mean(entropy)
                        entropy = []
                if real_done:
                    if not self.aux_dones[env_idx]:
                        self.aux_entropy = np.mean(entropy)
                        self.tar_entropy = 0.
                        self.controller.update(self.goals[env_idx], next_obs[env_idx],
                                               self.episode_step[env_idx]/self.max_episode_length,
                                               acer_step / self.total_steps)
                        self.mem = "aux_succ:False, real_step:{}".format(self.episode_step[env_idx])
                        aux_goal = one_hot_to_arr(self.aux_goal)
                        episode_info["aux_info"] = dict(aux_x=aux_goal[0],
                                                        aux_y=aux_goal[1],
                                                        succ=False)
                        self.aux_entropy = np.mean(entropy)
                    else:
                        self.tar_entropy = np.mean(entropy)
                    entropy = []

                    logger.info("aux_goals:{}, final:{}, {}".format(one_hot_to_arr(self.aux_goal),
                                                                    one_hot_to_arr(next_obs[env_idx]),
                                                                    self.mem))
                    self.aux_dones[env_idx] = False
                    self.aux_step[env_idx] = 0
                    self.goals[env_idx] = self.controller.step_goal()
                    self.aux_goal = self.goals[env_idx].copy()
                    self.episode_step[env_idx] = 0
                    self.episode[env_idx] += 1

                    episode_info["ent_info"] = dict(aux_ent=self.aux_entropy, tar_ent=self.tar_entropy)
                    logger.info("aux_ent:{:.4f}, tar_ent:{:.4f}".format(self.aux_entropy, self.tar_entropy))
                    # episode_return is correct because I do not edify reward fn.
                    episode_info["episode"] = infos[env_idx].get("episode")

            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
            mb_dones.append(dones)
        mb_obs.append(np.copy(self.obs))
        mb_masks.append(np.copy(self.dones))

        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)
        mb_goals = np.asarray(mb_goals, dtype=self.goals.dtype).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = np.asarray(mb_masks, dtype=np.bool).swapaxes(1, 0)
        mb_aux = np.asarray(mb_aux, dtype=np.bool).swapaxes(1, 0)

        index = np.where(mb_rewards.astype(int))
        if not np.array_equal(mb_goals[index], mb_next_obs[index]):
            raise ValueError
        for i in range(self.nsteps):
            if np.array_equal(mb_goals[0][i], mb_next_obs[0][i]):
                if mb_rewards[0][i] != 1.0:
                    raise ValueError("mb_goals:{}, mb_next_obs:{}, mb_rewards:{} should be 1.0".format(
                        mb_goals[0][i], mb_next_obs[0][i], mb_rewards[0][i]
                    ))
            else:
                if mb_rewards[0][i] + 0.1 / np.prod(self.maze_shape) > 1e-6:
                    raise ValueError("error:{}, index:{}, reward:{}".format(
                        mb_rewards[0][i] + 0.1 / np.prod(self.maze_shape), i, mb_rewards[0][i]))

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        results = dict(
            obs=mb_obs,
            next_obs=mb_next_obs,
            actions=mb_actions,
            rewards=mb_rewards,
            aux=mb_aux,
            mus=mb_mus,
            dones=mb_dones,
            masks=mb_masks,
            goal_obs=mb_goals,
            episode_info=episode_info,
        )
        return results

