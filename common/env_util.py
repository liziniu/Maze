import multiprocessing
import sys
import gym
from collections import defaultdict
from baselines.common.vec_env import VecNormalize
from baselines.common.tf_util import get_session
import re
import tensorflow as tf
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from common.atari_wrappers import wrap_deepmind, MaxAndSkipEnv, NoopResetEnv, TimeLimit
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env import DummyVecEnv
from baselines.common import retro_wrappers
import os
from baselines.common.vec_env import VecEnvWrapper
import numpy as np
from gym import spaces
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def get_env_type(env_id, env_type=None):
    if env_type is not None:
        return env_type, env_id
    if "Mario" in env_id:
        return "atari", env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def build_env(env_id, num_env, alg, reward_scale=1.0, env_type=None, gamestate=None, seed=None, prefix=""):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = num_env or ncpu

    env_type, env_id = get_env_type(env_id, env_type)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, prefix=prefix, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, prefix=prefix, seed=seed)
        else:
            frame_stack_size = 4
            if "SuperMarioBros" in env_id:
                wrapper_kwargs = {"episode_life": False}
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=gamestate, reward_scale=reward_scale,
                               prefix=prefix, wrapper_kwargs=wrapper_kwargs)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, num_env or 1, seed, reward_scale=reward_scale,
                           flatten_dict_observations=flatten_dict_observations, prefix=prefix)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 prefix=""):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            prefix=prefix,
            mpi_rank=mpi_rank,
            sub_rank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            logger_dir=logger_dir
        )

    set_global_seeds(seed)
    if num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_atari(env_id, max_episode_steps=None):
    if "SuperMario" in env_id:
        import gym_super_mario_bros
        from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = gym_super_mario_bros.make(env_id)
        env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    else:
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_env(env_id, env_type, prefix="", mpi_rank=0, sub_rank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, logger_dir=None):
    wrapper_kwargs = wrapper_kwargs or {}
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    else:
        if "maze" in env_id:
            max_episode_steps = wrapper_kwargs.get("max_episode_steps", None)
            env = make_maze(env_id, max_episode_steps)
        else:
            env = gym.make(env_id)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + sub_rank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(prefix) + str(mpi_rank) + '.' + str(sub_rank)),
                  allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        if 'frame_stack' not in wrapper_kwargs:
            wrapper_kwargs['frame_stack'] = 1
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env


def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env


def parser_env_id(env):
    env_id = None
    if hasattr(env, "spec"):
        env_id = env.spec.id
    elif hasattr(env, "venv"):
        try:
            env_id = env.venv.spec.id
        except:
            pass
    return env_id


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                # self.stackedobs = 0
                if obs[i].shape == self.stackedobs[i].shape:
                    self.stackedobs[i] = obs[i]
                else:
                    self.stackedobs[i, ..., :-obs.shape[-1]] = obs[i]
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        if obs.shape == self.stackedobs.shape:
            self.stackedobs = obs
        else:
            self.stackedobs[..., -self.nstack:] = obs
        return self.stackedobs


class MazeObservationSpace(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert len(size) == 2 and size[0] == size[1]
        self.size = size[0]
        gym.Space.__init__(self, (), np.int64)
        self.dtype = np.dtype(np.int64)
        self.shape = (len(size), size[0])
        self.ndim = len(size)

    def sample(self):
        one_hot_vector = np.zeros(self.shape)
        one_hot_vector[np.random.randint(0, self.size, self.ndim)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != self.ndim:
                return False
            else:
                return True
        else:
            return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size


class MazeWrapper(gym.core.Wrapper):
    def __init__(self, env):
        gym.core.Wrapper.__init__(self, env)
        self.observation_space = MazeObservationSpace(env.unwrapped.maze_size)

    def step(self, action):
        if hasattr(self, "_step"):
            self.step = self._step
            obs, reward, done, info = self.step(action)
        else:
            obs, reward, done, info = self.env.step(action)
        _obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        index = (np.arange(self.observation_space.ndim), obs.astype(int))
        _obs[index] = 1
        return _obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        _obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        index = (np.arange(self.observation_space.ndim), obs.astype(int))
        _obs[index] = 1
        return _obs


def make_maze(env_id, max_episode_steps):
    env = gym.make(env_id)
    grid = env_id.split("-")[2]
    size = np.prod([int(x) for x in grid.split("x")])
    if size == 100:
        max_episode_steps = 3000
        env = TimeLimit(env, max_episode_steps)
    return MazeWrapper(env)

if __name__ == "__main__":
    import gym_maze
    env = make_maze("maze-sample-10x10-v0", None)
    print(env.observation_space.shape)