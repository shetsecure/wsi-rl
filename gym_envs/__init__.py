import gymnasium as gym


gym.register(
    id="gym_envs/WSIWorldEnv-v1",
    entry_point="gym_envs.envs:WSIWorldEnv",
    max_episode_steps=10000,
    kwargs={"wsi_path": "/home/lipade/rl/test.ndpi"},
)
