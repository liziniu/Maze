import functools
import tensorflow as tf
from baselines import logger
from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.a2c.utils import batch_to_seq, seq_to_batch
from baselines.a2c.utils import cat_entropy_softmax
from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.utils import get_by_index, check_shape, avg_norm, q_explained_variance
from common.util import gradient_add
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np


# remove last step
def strip(var, nenvs, nsteps, flat=False):
    vars = batch_to_seq(var, nenvs, nsteps, flat)
    return seq_to_batch(vars, flat)


def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets;
    vs: nenv, nsteps (takes obs_{t+1} and g_t as inputs)

    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps, True)   # (by lizn, only the next state value)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        if i > 0:
            qret = (rho_bar[i] * (qret - q_is[i])) + vs[i-1]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret


# For ACER with PPO clipping instead of trust region
# def clip(ratio, eps_clip):
#     # assume 0 <= eps_clip <= 1
#     return tf.minimum(1 + eps_clip, tf.maximum(1 - eps_clip, ratio))


class Model(object):
    def __init__(self, sess, policy, ob_space, ac_space, nenvs, nsteps, ent_coef, q_coef, gamma,
                 max_grad_norm, lr, rprop_alpha, rprop_epsilon, total_timesteps, lrschedule, c, trust_region,
                 alpha, delta, scope, load_path, goal_shape):
        self.sess = sess
        self.nenv = nenvs
        self.nsteps = nsteps
        self.goal_shape = goal_shape

        nact = ac_space.n
        nbatch = nenvs * nsteps
        eps = 1e-6

        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.A = tf.placeholder(tf.int32, [nbatch], name="action")  # actions
            self.D = tf.placeholder(tf.float32, [nbatch], name="dones")  # dones
            self.R = tf.placeholder(tf.float32, [nbatch], name="rewards")  # rewards, not returns
            self.MU = tf.placeholder(tf.float32, [nbatch, nact], name="mus")  # mu's
            self.LR = tf.placeholder(tf.float32, [], name="lr")
            self.AUX = tf.placeholder(tf.float32, [nbatch], name="aux")

            self.V_NEXT = tf.placeholder(tf.float32, [nbatch], name="value_next")  # (by lzn: we revise goal-conditioned next value)

            step_ob_placeholder = tf.placeholder(ob_space.dtype, (nenvs,) + ob_space.shape, "step_ob")
            step_goal_placeholder = tf.placeholder(tf.float32, (nenvs,) + goal_shape, "step_goal")
            step_goal_encoded = step_goal_placeholder

            train_ob_placeholder = tf.placeholder(ob_space.dtype, (nenvs * nsteps,) + ob_space.shape, "train_ob")
            train_goal_placeholder = tf.placeholder(tf.float32, (nenvs * nsteps,) + goal_shape,
                                                    "train_goal")
            train_goal_encoded = train_goal_placeholder
            concat_on_latent = False
            self.step_model = policy(nbatch=nenvs, nsteps=1, observ_placeholder=step_ob_placeholder, sess=self.sess,
                                     goal_placeholder=step_goal_placeholder, concat_on_latent=concat_on_latent,
                                     goal_encoded=step_goal_encoded)
            self.train_model = policy(nbatch=nbatch, nsteps=nsteps, observ_placeholder=train_ob_placeholder,
                                      sess=self.sess,
                                      goal_placeholder=train_goal_placeholder, concat_on_latent=concat_on_latent,
                                      goal_encoded=train_goal_encoded)

        variables = find_trainable_variables
        self.params = params = variables(scope)
        logger.info("========================== {} =============================".format(scope))
        for var in params:
            logger.info(var)
        logger.info("========================== {} =============================\n".format(scope))

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        ema_apply_op = ema.apply(params)

        # print("========================== Ema =============================")

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            # print(v.name)
            return v

        # print("========================== Ema =============================")

        with tf.variable_scope(scope, custom_getter=custom_getter, reuse=True):
            self.polyak_model = policy(nbatch=nbatch, nsteps=nsteps, observ_placeholder=train_ob_placeholder,
                                       goal_placeholder=train_goal_placeholder, sess=self.sess,
                                       concat_on_latent=concat_on_latent, goal_encoded=train_goal_encoded)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i

        # action probability distributions according to self.train_model, self.polyak_model and self.step_model
        # poilcy.pi is probability distribution parameters; to obtain distribution that sums to 1 need to take softmax
        train_model_p = tf.nn.softmax(self.train_model.pi)
        polyak_model_p = tf.nn.softmax(self.polyak_model.pi)
        self.step_model_p = tf.nn.softmax(self.step_model.pi)
        # (todo by lizn, use this to calculate next value)
        v = self.v = tf.reduce_sum(train_model_p * self.train_model.q, axis=-1)  # shape is [nenvs * (nsteps)]

        # strip off last step
        # (todo by lizn, we don't need strip)
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model_p, polyak_model_p, self.train_model.q])
        # f, f_pol, q = map(lambda x: x, [train_model_p, polyak_model_p, self.train_model.q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, self.A)
        q_i = get_by_index(q, self.A)

        # Compute ratios for importance truncation
        rho = f / (self.MU + eps)
        rho_i = get_by_index(rho, self.A)

        # Calculate Q_retrace targets
        qret = q_retrace(self.R, self.D, q_i, self.V_NEXT, rho_i, nenvs, nsteps, gamma)  # (todo by lizn, use new next state value) = q_retrace()
        # Calculate losses
        # Entropy
        # entropy = tf.reduce_mean(strip(self.train_model.pd.entropy(), nenvs, nsteps))

        entropy = tf.reduce_mean(self.AUX * cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)  # (todo by lzn: we do not need the strip the last one)
        check_shape([qret, v, rho_i, f_i], [[nenvs * nsteps]] * 4)
        check_shape([rho, f, q], [[nenvs * nsteps, nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps)  # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[nenvs * nsteps, nact]] * 2)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f),
                                axis=1)  # IMP: This is sum, as expectation wrt f
        loss_bc = -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[nenvs * nsteps]] * 2)
        ev = q_explained_variance(tf.reshape(q_i, [nenvs, nsteps]), tf.reshape(qret, [nenvs, nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i) * 0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)

        # Goal loss
        loss = loss_policy + q_coef * loss_q - ent_coef * entropy

        if trust_region:
            g = tf.gradients(- (loss_policy - ent_coef * entropy) * nsteps * nenvs, f)  # [nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps)  # [nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) /
                             (tf.reduce_sum(tf.square(k), axis=-1) + eps))  # [nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = avg_norm(k)
            avg_norm_g = avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [nenvs * nsteps, 1]) * k
            grads_f = -g / (
                nenvs * nsteps)  # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(loss_q * q_coef, params)
            # print("=========================== gards add ==============================")
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]
            # print("=========================== gards add ==============================\n")
            avg_norm_grads_f = avg_norm(grads_f) * (nsteps * nenvs)
            norm_grads_q = tf.global_norm(grads_q)
            norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        # trainer = tf.train.AdamOptimizer(learning_rate=self.LR)
        _policy_opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_policy_opt_op]):
            _train_policy = tf.group(ema_apply_op)

        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        self.run_ops_policy = [_train_policy, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        self.names_ops_policy = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
                                 'norm_grads']
        if trust_region:
            self.run_ops_policy = self.run_ops_policy + [
                norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g, avg_norm_k_dot_g,
                avg_norm_adj]
            self.names_ops_policy = self.names_ops_policy + [
                'norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k', 'avg_norm_g', 'avg_norm_k_dot_g',
                'avg_norm_adj']
        self.names_ops_policy = [scope + "_" + x for x in self.names_ops_policy]  # scope as prefix

        self.save = functools.partial(save_variables, sess=self.sess, variables=params)
        self.load = functools.partial(load_variables, sess=self.sess, variables=params)

        self.initial_state = self.step_model.initial_state
        if load_path is not None:
            tf.global_variables_initializer().run(session=self.sess)
            logger.info("loading pretrained model from {}".format(load_path))
            self.load(load_path)
        else:
            tf.global_variables_initializer().run(session=self.sess)

    def train_policy(self, obs, next_obs, actions, rewards, dones, mus, states, masks, steps, goal_obs, aux,
                     verbose=False):
        cur_lr = self.lr.value_steps(steps)
        # 1. calculate v_{t+1} using obs_{t+1} and g_t
        td_map = {self.train_model.X: next_obs}
        assert hasattr(self.train_model, "goals")
        td_map[self.train_model.goals] = goal_obs
        v_next = self.sess.run(self.v, feed_dict=td_map)
        # 2. use obs_t, goal_t, v_{t+1} to train policy
        td_map = {self.train_model.X: obs, self.polyak_model.X: obs, self.A: actions, self.R: rewards, self.D: dones,
                  self.MU: mus, self.LR: cur_lr, self.V_NEXT: v_next, self.AUX: aux}

        ##########################################
        debug = False
        if debug:
            self._test(obs, next_obs, actions, rewards, dones, mus, goal_obs )
        ############################################
        assert hasattr(self.train_model, "goals")
        assert hasattr(self.polyak_model, "goals")
        if hasattr(self, "goal_rms"):
            self.goal_rms.update(goal_obs)
        td_map[self.train_model.goals] = goal_obs
        td_map[self.polyak_model.goals] = goal_obs
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
            td_map[self.polyak_model.S] = states
            td_map[self.polyak_model.M] = masks
        if verbose:
            names_ops_policy = self.names_ops_policy.copy()
            values_ops_policy = self.sess.run(self.run_ops_policy, td_map)[1:]  # strip off _train
        else:
            names_ops_policy = self.names_ops_policy.copy()[:8]  # not including trust region
            values_ops_policy = self.sess.run(self.run_ops_policy, td_map)[1:][:8]

        return names_ops_policy, values_ops_policy

    def step(self, observation, **kwargs):
        return self.step_model.evaluate([self.step_model.action, self.step_model_p, self.step_model.state],
                                        observation, **kwargs)

    def _test(self, obs, next_obs, actions, rewards, dones, mus, goal_obs):
        _obs, _next_obs, _actions, _dones, _goals, _mus, _rewards = self.generate_fake(obs, next_obs, actions, dones,
                                                                                       goal_obs, mus, rewards)
        td_map = dict()
        td_map[self.train_model.goals] = _goals
        td_map[self.train_model.X] = _next_obs
        v_next = self.sess.run(self.v, feed_dict=td_map)
        print("v_next", v_next)
        td_map[self.train_model.X] = _obs
        td_map[self.A] = _actions
        td_map[self.R] = _rewards
        td_map[self.MU] = _mus
        td_map[self.D] = _dones
        td_map[self.V_NEXT] = v_next
        print("------td map--------")
        print(td_map)
        print("------td map--------")
        print("-------q_iter--------")
        q_iter = self.sess.run(self.q_iter, feed_dict=td_map)
        print(q_iter)
        print("-------q_iter--------")
        print("-------q_iter_after--------")
        q_iter_after = self.sess.run(self.q_iter, feed_dict=td_map)
        print(q_iter_after)
        print("-------q_iter_after--------")
        print("--------rs---------")
        rs = self.sess.run(self.rs, feed_dict=td_map)
        print(rs)
        print("--------rs---------")
        q_i, rho_i, qret = self.sess.run([self.q_i, self.rho_i, self.qret], feed_dict=td_map)
        print("q_i", q_i)
        print("rho_i", rho_i)
        print("q_ret", qret)
        assert 0

    def generate_fake(self, obs, next_obs, actions, dones, goals, mus, rewards):
        obs_new = np.random.randn(self.nenv, self.nsteps+1, *obs.shape[1:])
        _obs = obs_new[:, :-1].reshape((-1, ) + obs.shape[1:])
        _next_obs = obs_new[:, 1:].reshape((-1, ) + next_obs.shape[1:])
        _actions = np.ones_like(actions)
        _dones = dones
        _goals = np.zeros_like(goals)
        _mus = np.random.randn(*mus.shape)
        _mus = _mus / np.sum(_mus, axis=-1, keepdims=True)
        print(self.sess.run(self.params))
        print("obs", obs)
        print("_obs", obs_new)
        _rewards = np.ones_like(rewards)
        return _obs, _next_obs, _actions, _dones, _goals, _mus, _rewards