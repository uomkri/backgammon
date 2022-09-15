from gym.envs.registration import register

register(
    id='backgammon-v1',
    entry_point='gym_backgammon.envs:BackgammonEnv',
)