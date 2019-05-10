import numpy as np
from queue import PriorityQueue
import sys
from her.util import vf_dist


class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, env, sample_goal_fn, reward_fn, nsteps, goal_shape, her, size=50000):
        self.nenv = env.num_envs
        self.nsteps = nsteps
        assert callable(sample_goal_fn)
        assert callable(reward_fn)
        self.sample_goal_fn = sample_goal_fn

        self.reward_fn = reward_fn
        # self.nh, self.nw, self.nc = env.observation_space.shape
        self.obs_shape = env.observation_space.shape
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.goal_shape = goal_shape

        self.nbatch = self.nenv * self.nsteps
        self.size = size // self.nsteps  # Each loc contains nenv * nsteps frames, thus total buffer is nenv * size frames

        self.maze_size = np.prod([int(x) for x in env.spec.id.split("-")[2].split("x")])
        self.her_gain = 0.
        # Memory
        self.obs = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None
        self.masks = None
        self.goals = None
        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

        self.her = her

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.num_in_buffer >= (frames // self.nsteps)

    def can_sample(self):
        return self.num_in_buffer > 0

    # Generate stacked frames
    def decode(self, enc_obs, dones):
        # enc_obs has shape [nenvs, nsteps + nstack, nh, nw, nc]
        # dones has shape [nenvs, nsteps]
        # returns stacked obs of shape [nenv, (nsteps + 1), nh, nw, nstack*nc]

        return _stack_obs(enc_obs, dones,
                          nsteps=self.nsteps)

    def put(self, episode_batch):
        # enc_obs [nenv, (nsteps + nstack), nh, nw, nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]
        obs, actions, rewards, mus, dones, masks, goal_obs = \
            episode_batch["obs"], episode_batch["actions"], episode_batch["rewards"], episode_batch["mus"],\
            episode_batch["dones"], episode_batch["masks"], episode_batch["goal_obs"],

        if self.obs is None:
            self.obs = np.empty([self.size] + list(obs.shape), dtype=self.obs_dtype)
            self.actions = np.empty([self.size] + list(actions.shape), dtype=self.ac_dtype)
            self.rewards = np.empty([self.size] + list(rewards.shape), dtype=np.float32)
            self.mus = np.empty([self.size] + list(mus.shape), dtype=np.float32)
            self.dones = np.empty([self.size] + list(dones.shape), dtype=np.bool)
            self.masks = np.empty([self.size] + list(masks.shape), dtype=np.bool)
            self.goals = np.empty([self.size] + list(goal_obs.shape), dtype=self.obs_dtype)

        self.obs[self.next_idx] = obs
        self.actions[self.next_idx] = actions
        self.rewards[self.next_idx] = rewards
        self.mus[self.next_idx] = mus
        self.dones[self.next_idx] = dones
        self.masks[self.next_idx] = masks
        self.goals[self.next_idx] = goal_obs
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, x, idx, envx):
        nenv = self.nenv
        out = np.empty([nenv] + list(x.shape[2:]), dtype=x.dtype)
        for i in range(nenv):
            out[i] = x[idx[i], envx[i]]
        return out

    # todo: add her
    def get(self, **kwargs):
        # returns
        # obs [nenv, (nsteps + 1), nh, nw, nstack*nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]
        nenv = self.nenv
        assert self.can_sample()

        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, nenv)
        envx = np.arange(nenv)

        take = lambda x: self.take(x, idx, envx)  # for i in range(nenv)], axis = 0)
        dones = take(self.dones)

        obs = take(self.obs)  # (nenv, nstep+1, nh, nw, nc)
        actions = take(self.actions)
        rewards = take(self.rewards)
        mus = take(self.mus)
        masks = take(self.masks)

        results = dict(obs=obs, actions=actions, rewards=rewards, mus=mus, dones=dones, masks=masks)
        goal_obs = take(self.goals)  # (nenv, nstep, nh, nw, nc)

        if self.her:
            her_idx, future_idx = self.sample_goal_fn(dones)
            goal_obs[her_idx] = obs[:, 1:][future_idx]
            new_rewards = self.reward_fn(obs[:, 1:], goal_obs, self.maze_size)
            results["rewards"] = new_rewards
            self.her_gain = new_rewards - rewards
        results["her_gain"] = self.her_gain
        results["goal_obs"] = goal_obs

        return results

    @property
    def memory_usage(self):
        if self.obs is None:
            return 0.
        else:
            usage = sys.getsizeof(self.obs) + sys.getsizeof(self.goals) + sys.getsizeof(self.actions) * 5
            return usage // (1024 ** 3)


def _stack_obs_ref(enc_obs, dones, nsteps):
    nenv = enc_obs.shape[0]
    nstack = enc_obs.shape[1] - nsteps
    nh, nw, nc = enc_obs.shape[2:]
    obs_dtype = enc_obs.dtype
    obs_shape = (nh, nw, nc * nstack)

    mask = np.empty([nsteps + nstack - 1, nenv, 1, 1, 1], dtype=np.float32)
    obs = np.zeros([nstack, nsteps + nstack, nenv, nh, nw, nc], dtype=obs_dtype)
    x = np.reshape(enc_obs, [nenv, nsteps + nstack, nh, nw, nc]).swapaxes(1, 0)  # [nsteps + nstack, nenv, nh, nw, nc]

    mask[nstack - 1:] = np.reshape(1.0 - dones, [nenv, nsteps, 1, 1, 1]).swapaxes(1, 0)  # keep
    mask[:nstack - 1] = 1.0

    # y = np.reshape(1 - dones, [nenvs, nsteps, 1, 1, 1])
    for i in range(nstack):
        obs[-(i + 1), i:] = x
        # obs[:,i:,:,:,-(i+1),:] = x
        x = x[:-1] * mask
        mask = mask[1:]

    return np.reshape(obs[:, (nstack - 1):].transpose((2, 1, 3, 4, 0, 5)), (nenv, (nsteps + 1)) + obs_shape)


def _stack_obs(enc_obs, dones, nsteps):
    nenv = enc_obs.shape[0]
    nstack = enc_obs.shape[1] - nsteps
    nc = enc_obs.shape[-1]

    obs_ = np.zeros((nenv, nsteps + 1) + enc_obs.shape[2:-1] + (enc_obs.shape[-1] * nstack,), dtype=enc_obs.dtype)
    mask = np.ones((nenv, nsteps + 1), dtype=enc_obs.dtype)
    mask[:, 1:] = 1.0 - dones
    mask = mask.reshape(mask.shape + tuple(np.ones(len(enc_obs.shape) - 2, dtype=np.uint8)))

    for i in range(nstack - 1, -1, -1):
        obs_[..., i * nc: (i + 1) * nc] = enc_obs[:, i: i + nsteps + 1, :]
        if i < nstack - 1:
            obs_[..., i * nc: (i + 1) * nc] *= mask
            mask[:, 1:, ...] *= mask[:, :-1, ...]

    return obs_


def test_stack_obs():
    nstack = 7
    nenv = 1
    nsteps = 5

    obs_shape = (2, 3, nstack)

    enc_obs_shape = (nenv, nsteps + nstack) + obs_shape[:-1] + (1,)
    enc_obs = np.random.random(enc_obs_shape)
    dones = np.random.randint(low=0, high=2, size=(nenv, nsteps))

    stacked_obs_ref = _stack_obs_ref(enc_obs, dones, nsteps=nsteps)
    stacked_obs_test = _stack_obs(enc_obs, dones, nsteps=nsteps)

    np.testing.assert_allclose(stacked_obs_ref, stacked_obs_test)
