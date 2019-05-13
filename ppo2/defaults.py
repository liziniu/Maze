def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )

def retro():
    return atari()


def maze(env_id):
    if "5x5" in env_id:
        return dict(
            nsteps=128, nminibatches=4,
            lam=0.95, gamma=0.99, noptepochs=4, log_interval=100,
            ent_coef=.05,
            lr=lambda f: f * 5e-4,
            cliprange=0.2,
        )
    elif "10x10" in env_id:
        return dict(
            nsteps=2048, nminibatches=32,
            lam=0.95, gamma=0.99, noptepochs=4, log_interval=2,
            ent_coef=.05,
            lr=lambda f: f * 5e-4,
            cliprange=0.2,
        )
    else:
        raise ValueError("env_id:{} wrong.".format(env_id))