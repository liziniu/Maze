import threading
import numpy as np
import sys
from her2.util import vf_dist
from baselines import logger
from her3.meta_controller import one_hot_to_arr, arr_to_one_hot


def check_reward_fn(next_obs, dones, maze_size):
    nenv, nsteps = next_obs.shape[0], next_obs.shape[1]
    rewards = np.empty([nenv, nsteps], dtype=np.float32)
    for i in range(nenv):
        for j in range(nsteps):
            if np.array_equal(next_obs[i][j], np.array([0., 0.])) and dones[i][j]:
                rewards[i][j] = 1.0
            else:
                rewards[i][j] = -0.1 / maze_size
    return rewards


class ReplayBuffer:
    def __init__(self, env, sample_goal_fn, reward_fn, nsteps, size, keys, her, revise_done):
        """Creates a replay buffer.

        Args:
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        nenv = self.nenv = env.num_envs
        self.nsteps = nsteps
        self.size = size // self.nsteps
        self.sample_goal_fn = sample_goal_fn
        self.reward_fn = reward_fn

        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.keys = keys
        # self._trajectory_buffer = Trajectory(nenv, keys)
        self.buffers = [{key: None for key in keys} for _ in range(nenv)]
        self._cache = [{} for _ in range(self.nenv)]

        self.her = her
        self.her_gain = 0.
        self.revise_done = revise_done
        self.maze_size = np.prod([int(x) for x in env.spec.id.split("-")[2].split("x")])
        self.true_goal = np.array([int(x) for x in env.spec.id.split("-")[2].split("x")]) -1
        self.true_goal = arr_to_one_hot(self.true_goal, self.true_goal[0]+1)
        # memory management
        self.lock = threading.Lock()
        self.current_size = 0   # num of sub-trajectories rather than transitions

    def get(self, use_cache, downsample=True):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        samples = {key: [] for key in self.keys}
        samples["her_gain"] = 0.
        if not use_cache:
            cache = [{} for _ in range(self.nenv)]
            for i in range(self.nenv):
                if downsample:
                    interval = 100     # 20 sub-trajectories
                    if self.current_size < interval:
                        start, end = 0, self.current_size
                    else:
                        start = np.random.randint(0, max(self.current_size-interval, self.current_size))
                        end = min(self.current_size, start+interval)
                else:
                    start, end = 0, self.current_size
                for key in self.keys:
                    if key in ["obs", "masks"]:
                        cache[i][key] = self.buffers[i][key][start*(self.nsteps+1):end*(self.nsteps+1)].copy()
                    else:
                        cache[i][key] = self.buffers[i][key][start*self.nsteps:end*self.nsteps].copy()
            if self.her:
                for i in range(self.nenv):
                    dones = cache[i]["dones"]
                    deaths = cache[i]["deaths"]
                    her_index, future_index = self.sample_goal_fn(dones, deaths)
                    reach_rewards = self.reward_fn(cache[i]["next_obs"][None, :], cache[i]["goal_obs"][None, :], self.maze_size)
                    reach_rewards = reach_rewards.flatten()
                    reach_index = np.where(reach_rewards.astype(int))
                    error = np.sum(np.abs(cache[i]["rewards"][reach_index] - reach_rewards[reach_index]))
                    assert error < 1e-6, "error:{}".format(error)
                    cache[i]["goal_obs"][her_index] = cache[i]["next_obs"][future_index]
            self._cache = cache.copy()
        else:
            cache = self._cache.copy()

        for i in range(self.nenv):
            transitions = cache[i]
            real_size = len(transitions["obs"]) // (self.nsteps + 1)
            index = np.random.randint(0, real_size)
            for key in self.keys:
                if key in ["obs", "masks"]:
                    start, end = index*(self.nsteps+1), (index+1)*(self.nsteps+1)
                else:
                    start, end = index*self.nsteps, (index+1)*self.nsteps
                samples[key].append(transitions[key][start:end])

        for key in self.keys:
            samples[key] = np.asarray(samples[key])
        if self.her:
            rewards = samples["rewards"]
            reach_rewards = self.reward_fn(samples["next_obs"], samples["goal_obs"], self.maze_size)

            reach_index = np.where(reach_rewards.astype(int))
            new_rewards = np.copy(rewards)
            new_rewards[reach_index] = 1.0

            dead_index = np.where(samples["deaths"])
            for k in range(len(dead_index[0])):
                env_index = dead_index[0][k]
                step = dead_index[1][k]
                assert np.array_equal(samples["goal_obs"][env_index][step], self.true_goal) or \
                       np.array_equal(samples["goal_obs"][env_index][step], samples["next_obs"][env_index][step])

            if self.revise_done:
                samples["dones"][reach_index] = True

            samples["her_gain"] = np.mean(new_rewards) - np.mean(rewards)
            samples["rewards"] = new_rewards
            if samples["her_gain"] < 0:
                raise ValueError("her gain:{} should be large than 0.".format(samples["her_gain"]))
        return samples

    def put(self, episode_batch):
        """episode_batch: dict of data. (nenv, nsteps, feature_shape)

        """
        assert isinstance(episode_batch, dict)
        key = self.keys[0]
        nenv, steps = episode_batch[key].shape[:2]
        assert nenv == self.nenv
        for i in range(nenv):
            for key in self.keys:
                x = episode_batch[key][i]
                if self.buffers[i][key] is None:
                    if key in ["obs", "masks"]:
                        maxlen = self.size * (self.nsteps + 1)
                    else:
                        maxlen = self.size * self.nsteps
                    self.buffers[i][key] = np.empty((maxlen, ) + x.shape[1:], dtype=x.dtype)
                if key in ["obs", "masks"]:
                    start, end = self.current_size*(self.nsteps+1), (self.current_size+1)*(self.nsteps+1)
                    self.buffers[i][key][start:end] = x
                else:
                    start, end = self.current_size*self.nsteps, (self.current_size+1)*self.nsteps
                    self.buffers[i][key][start:end] = x
        self.current_size += 1
        self.current_size %= self.size

        debug = False
        if debug:
            self.check_put()

    @property
    def memory_usage(self):
        usage = 0
        for key in self.keys:
            usage += sys.getsizeof(self.buffers[0][key])
        return (usage * 2) // (1024 ** 3)

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.current_size * self.nsteps >= frames

    def check_put(self):
        for i in range(self.nenv):
            start, end = self.current_size-1, self.current_size
            if end == 0:
                start, end = self.size-1, self.size
            dones = decode(self.buffers[i]["dones"][start*self.nsteps:end*self.nsteps], self.nsteps)
            next_obs = decode(self.buffers[i]["obs"][start*(self.nsteps+1):end*(self.nsteps+1)], self.nsteps+1)
            rewards = decode(self.buffers[i]["rewards"][start*self.nsteps:end*self.nsteps], self.nsteps).astype(int)

            done_index = np.where(dones)
            next_index = (done_index[0], np.array(done_index[1])+1)
            obs_selected = next_obs[next_index]
            for o in obs_selected:
                assert np.array_equal(o, np.array([0., 0.])), "o:{}".format(o)

            done_index = np.where(rewards)
            next_index = (done_index[0], np.array(done_index[1])+1)
            obs_selected = next_obs[next_index]
            for o in obs_selected:
                assert np.array_equal(o, np.array([0., 0.])), "o:{}".format(o)


def decode_obs(enc_obs, nsteps):
    assert len(enc_obs) % (nsteps+1) == 0
    nb_segment = len(enc_obs) // (nsteps + 1)
    segments = np.split(enc_obs, nb_segment)
    new_arr = []
    for sub_arr in segments:
        new_arr.append(sub_arr[1:])
    new_arr = np.concatenate(new_arr, axis=0)
    return new_arr

def decode(x, nsteps):
    assert len(x) % nsteps== 0
    nb_segment = len(x) // nsteps
    segments = np.split(x, nb_segment)
    new_arr = []
    for sub_arr in segments:
        new_arr.append(sub_arr)
    new_arr = np.asarray(new_arr)
    return new_arr

if __name__ == "__main__":
    nsteps = 10
    enc_obs = []
    for j in range(2):
        for i in range(nsteps+1):
            enc_obs.append(j*10 + i)
    enc_obs = np.array(enc_obs)
    obs = decode_obs(enc_obs, nsteps)
    print(obs)