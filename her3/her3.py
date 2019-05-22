import time
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from acer.policies import build_policy
from her3.runner2 import Runner
from her3.runner3 import EvalRunner
from common.her_sample import make_sample_her_transitions
from her3.model import Model
from her3.util import Acer
from baselines.common.tf_util import get_session
import sys
import os
from her3.buffer2 import ReplayBuffer
from her3.defaults import get_store_keys


def learn(network, env, seed=None, nsteps=20, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=200, buffer_size=50000, replay_ratio=4, replay_start=10000, c=10.0, trust_region=True,
          alpha=0.99, delta=1, replay_k=1, env_eval=None, eval_interval=300, save_model=False, her=False,
          goal_shape=None, nb_train_epoch=4, save_interval=0, load_path=None, **network_kwargs):

    '''
    Main entrypoint for ACER (Actor-Critic with Experience Replay) algorithm (https://arxiv.org/pdf/1611.01224.pdf)
    Train an agent with given network architecture on a given environment using ACER.

    Parameters:
    ----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies

    env:                environment. Needs to be vectorized for parallel environment simulation.
                        The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel) (default: 20)

    nstack:             int, size of the frame stack, i.e. number of the frames passed to the step model. Frames are stacked along channel dimension
                        (last image dimension) (default: 4)

    total_timesteps:    int, number of timesteps (i.e. number of actions taken in the environment) (default: 80M)

    q_coef:             float, value function loss coefficient in the optimization objective (analog of vf_coef for other actor-critic methods)

    ent_coef:           float, policy entropy coefficient in the optimization objective (default: 0.01)

    max_grad_norm:      float, gradient norm clipping coefficient. If set to None, no clipping. (default: 10),

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    rprop_epsilon:      float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    rprop_alpha:        float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting factor (default: 0.99)

    log_interval:       int, number of updates between logging events (default: 100)

    buffer_size:        int, size of the replay buffer (default: 50k)

    replay_ratio:       int, now many (on average) batches of data to sample from the replay buffer take after batch from the environment (default: 4)

    replay_start:       int, the sampling from the replay buffer does not start until replay buffer has at least that many samples (default: 10k)

    c:                  float, importance weight clipping factor (default: 10)

    trust_region        bool, whether or not algorithms estimates the gradient KL divergence between the old and updated policy and uses it to determine step size  (default: True)

    delta:              float, max KL divergence between the old policy and updated policy (default: 1)

    alpha:              float, momentum factor in the Polyak (exponential moving average) averaging of the model parameters (default: 0.99)

    load_path:          str, path to load the model from (default: None)

    **network_kwargs:               keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                    For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    logger.info("Running Acer with following kwargs")
    logger.info(locals())
    logger.info("\n")

    set_global_seeds(seed)

    if env_eval is None:
        raise ValueError("env_eval is required!")

    network_kwargs = {"layer_norm": True}
    policy = build_policy(env, network, estimate_q=True, **network_kwargs)
    nenvs = env.num_envs
    nenvs_eval = env_eval.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    sess = get_session()
    if goal_shape is None:
        goal_shape = env.observation_space.shape

    model = Model(
        sess=sess, policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
        q_coef=q_coef, gamma=gamma, max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha,
        rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
        trust_region=trust_region, alpha=alpha, delta=delta, scope="her", goal_shape=goal_shape, load_path=load_path)

    def reward_fn(next_obs, goal_obs, maze_size):
        nenv, nsteps = next_obs.shape[0], next_obs.shape[1]
        rewards = np.empty([nenv, nsteps], dtype=np.float32)
        for i in range(nenv):
            for j in range(nsteps):
                if np.array_equal(next_obs[i][j], goal_obs[i][j]):
                    rewards[i][j] = 1.0
                else:
                    rewards[i][j] = -0.1 / maze_size
        return rewards

    # we still need two runner to avoid one reset others' envs.
    runner = Runner(env=env, model=model, nsteps=nsteps, save_interval=save_interval, total_steps=total_timesteps, her=her)
    runner_eval = EvalRunner(env=env_eval, model=model)

    if replay_ratio > 0:
        assert env.num_envs == env_eval.num_envs
        her_sample_fn = make_sample_her_transitions("future", replay_k)
        buffer = ReplayBuffer(env=env, nsteps=nsteps, size=buffer_size, keys=get_store_keys(),
                              reward_fn=reward_fn, sample_goal_fn=her_sample_fn, her=her)
    else:
        buffer = None
    acer = Acer(runner, model, buffer, log_interval,)
    acer.tstart = time.time()

    # === init to make sure we can get goal ===
    onpolicy_cnt = 0
    debug = False
    if debug:
        while True:
            runner.run(debug=True)

    while acer.steps < total_timesteps:
        acer.call(replay_start=replay_start, nb_train_epoch=nb_train_epoch)
        acer.steps += nenvs * nsteps
        onpolicy_cnt += 1
        if onpolicy_cnt % 30 == 0:
            acer.evaluate(runner_eval)
    acer.save(os.path.join(logger.get_dir(), "models", "{}.pkl".format(acer.steps)))

    return model