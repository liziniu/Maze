import datetime
import multiprocessing
import os.path as osp
import re
import sys
from collections import defaultdict
from importlib import import_module
import gym
import numpy as np
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from common.cmd_util import common_arg_parser, parse_unknown_args, parse_acer_mode
from common.env_util import build_env
import gym_maze
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
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


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type, env_id)
    alg_kwargs.update(extra_args)

    env = build_env(env_id=args.env, num_env=args.num_env, reward_scale=args.reward_scale, env_type=args.env_type,
                    gamestate=args.gamestate, seed=args.seed, alg=args.alg)

    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
    env_eval = build_env(env_id=args.env, num_env=args.num_env, reward_scale=args.reward_scale, env_type=args.env_type,
                         gamestate=args.gamestate, seed=args.seed, alg=args.alg, prefix="evaluation")
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if args.alg == "acer":
        use_expl_collect, use_eval_collect, use_random_policy_expl, dyna_source_list = parse_acer_mode(args.mode)

        alg_kwargs["use_expl_collect"] = use_expl_collect
        alg_kwargs["use_eval_collect"] = use_eval_collect
        alg_kwargs["use_random_policy_expl"] = use_random_policy_expl
        alg_kwargs["dyna_source_list"] = dyna_source_list
        alg_kwargs["store_data"] = args.store_data
        alg_kwargs["aux_task"] = args.aux_task

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        env_eval=env_eval,
        **alg_kwargs
    )

    return model, env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

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


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    # try:
    #     # first try to import the alg module from baselines
    #     alg_module = import_module('.'.join(['baselines', alg, submodule]))
    # except ImportError:
    #     # then from rl_algs
    #     alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    alg_module = import_module('.'.join([alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type, env_id):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)(env_id)
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def main(args, extra_args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    import os
    import shutil
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    path = osp.join("logs",  datetime.datetime.now().strftime("{}-{}-%Y-%m-%d-%H-%M-%S-%f".format(args.alg, args.env)))
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=path)
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    shutil.copytree(os.path.join(os.getcwd(), args.alg), os.path.join(path, "code", args.alg))
    shutil.copytree(os.path.join(os.getcwd(), "common"), os.path.join(path, "code", "common"))
    logger.info("cmd args:{}".format(args.__dict__))
    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()

    env.close()

    return model

if __name__ == '__main__':
    from multiprocessing import Process
    from copy import deepcopy
    list_p = []
    args = sys.argv
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    for i in range(args.num_exp):
        arg_i = deepcopy(args)
        arg_i.seed = args.seed + i
        p = Process(target=main, args=(arg_i, extra_args))
        list_p.append(p)
    for i, p in enumerate(list_p):
        p.start()
        print("Process:{} start".format(i))
    for i, p in enumerate(list_p):
        p.join()
        print("Process:{} end".format(i))
