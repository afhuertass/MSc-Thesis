from gym.envs.registration import register
register(id='Frankfurt-v0',
    entry_point='envs.frankfurt_env_dir:FrankfurtEnv'
)