import time
import numpy as np
from baselines import logger
from common.util import EpisodeStats
from copy import deepcopy
from baselines.common.tf_util import save_variables
import functools
import os
import sys
from common.util import DataRecorder


class Acer:
    def __init__(self, runner, model, buffer, log_interval):
        self.runner = runner
        self.model = model
        self.buffer = buffer
        self.log_interval = log_interval
        self.tstart = None
        self.keys = ["episode_return", "episode_length",  "rewards", "aux_x", "aux_y", "aux_succ_ratio"]
        self.keys += ["tar_ent", "aux_ent"]
        self.episode_stats = EpisodeStats(maxlen=10, keys=self.keys)
        self.steps = 0
        self.save_interval = self.runner.save_interval
        self.recoder = DataRecorder(os.path.join(logger.get_dir(), "samples"))

        sess = self.model.sess
        self.save = functools.partial(save_variables, sess=sess, variables=self.model.params)

    def call(self, replay_start, nb_train_epoch):
        runner, model, buffer, steps = self.runner, self.model, self.buffer, self.steps

        results = runner.run(acer_step=self.steps)
        buffer.put(results)

        self.record_episode_info(results["episode_info"])
        obs, next_obs, actions, rewards, mus, dones, masks, goal_obs, aux = self.adjust_shape(results)
        names_ops, values_ops = model.train_policy(
            obs, next_obs, actions, rewards, dones, mus, model.initial_state, masks, steps, goal_obs, aux)

        self.episode_stats.feed(np.mean(rewards), "rewards")
        if buffer.has_atleast(replay_start):
            for i in range(nb_train_epoch):
                if i == 0:
                    results = buffer.get(use_cache=False)
                else:
                    results = buffer.get(use_cache=True)
                obs, next_obs, actions, rewards, mus, dones, masks, goal_obs, aux = self.adjust_shape(results)
                names_ops, values_ops = model.train_policy(
                    obs, next_obs, actions, rewards, dones, mus, model.initial_state, masks, steps, goal_obs, aux)
                self.episode_stats.feed(np.mean(rewards), "rewards")

        if int(steps/runner.nbatch) % self.log_interval == 0:
            names_ops, values_ops = names_ops + ["memory_usage(GB)"], values_ops + [self.buffer.memory_usage]
            self.log(names_ops, values_ops)

            if int(steps/runner.nbatch) % (self.log_interval * 200) == 0:
                self.save(os.path.join(logger.get_dir(), "{}.pkl".format(self.steps)))

        if self.save_interval > 0 and int(steps / runner.nbatch) % self.save_interval == 0:
            results["acer_steps"] = self.steps
            self.recoder.store(results)
            self.recoder.dump()

    def adjust_shape(self, results):
        runner = self.runner

        obs = results["obs"][:, :-1].copy()
        # next_obs = results["obs"][:, 1:].copy()
        next_obs = results["next_obs"].copy()
        obs = obs.reshape((runner.nbatch, ) + runner.obs_shape)
        next_obs = next_obs.reshape((runner.nbatch, ) + runner.obs_shape)

        actions = results["actions"].reshape(runner.nbatch)
        rewards = results["rewards"].reshape(runner.nbatch)
        mus = results["mus"].reshape([runner.nbatch, runner.nact])
        dones = results["dones"].reshape([runner.nbatch])
        masks = results["masks"].reshape([runner.batch_ob_shape[0]])
        goal_obs = results["goal_obs"].reshape((runner.nbatch, ) + runner.obs_shape)
        aux = results["aux"].reshape(runner.nbatch)
        return obs, next_obs, actions, rewards, mus, dones, masks, goal_obs, aux

    def record_episode_info(self, episode_info):
        returns = episode_info.get("episode", None)
        if returns:
            self.episode_stats.feed(returns["r"], "episode_return")
            self.episode_stats.feed(returns["l"], "episode_length")
        aux_info = episode_info.get("aux_info", None)
        if aux_info:
            self.episode_stats.feed(aux_info["aux_x"], "aux_x")
            self.episode_stats.feed(aux_info["aux_y"], "aux_y")
            self.episode_stats.feed(aux_info["succ"], "aux_succ_ratio")
        ent_info = episode_info.get("ent_info", None)
        if ent_info:
            self.episode_stats.feed(ent_info["aux_ent"], "aux_ent")
            tar_ent = ent_info["tar_ent"]
            if tar_ent > 0:
                self.episode_stats.feed(ent_info["tar_ent"], "tar_ent")

    def log(self, names_ops, values_ops):
        logger.record_tabular("total_timesteps", self.steps)
        logger.record_tabular("fps", int(self.steps / (time.time() - self.tstart)))
        for name, val in zip(names_ops, values_ops):
            logger.record_tabular(name, float(val))
        for key in self.keys:
            if key in ["real_step", "aux_x", "aux_y"]:
                logger.record_tabular(key, self.episode_stats.get_last(key))
            else:
                logger.record_tabular(key, self.episode_stats.get_mean(key))
        logger.dump_tabular()


def f_dist(current_pos, goal_pos):
    dist = abs(float(current_pos["x_pos"]) - float(goal_pos["x_pos"])) + \
           abs(float(current_pos["y_pos"]) - float(goal_pos["y_pos"]))
    return dist

vf_dist = np.vectorize(f_dist)