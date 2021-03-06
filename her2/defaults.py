
THRESHOLD = 6


def atari():
    return dict(
        lrschedule='constant',
        load_path='data/goal_data_303_2503.pkl',
        nsteps=50,
        nb_train_epoch=8,
        desired_x_pos=500,
    )


def get_store_keys():
    return ["obs", "next_obs", "actions", "rewards", "mus", "dones",
            "masks", "goal_obs", "deaths"]
