python run.py \
    --env maze-sample-5x5-v0 \
    --env_type maze \
    --num_env 1 \
    --alg her2 \
    --num_timesteps 1e5 \
    --gpu 12,13,2 \
    --her 1.0 \
    --buffer2 1.0 \
    --replay_k 1.0 \
    --revise_done 0.0