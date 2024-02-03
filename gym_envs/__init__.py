import gymnasium as gym


gym.register(
    id="gym_envs/WSIEnv-v1",
    entry_point="gym_envs.envs:WSIEnv",
    max_episode_steps=10000,
    kwargs={"wsi_path": "/home/lipade/rl/test.ndpi"},
)


gym.register(
    id="WSIWorldEnv-v1",
    entry_point="gym_envs.envs:WSIWorldEnv",
    max_episode_steps=1000,
)

gym.register(
    id="WSIWorldEnv-v2",
    entry_point="gym_envs.envs:WSIWorldEnvV2",
    max_episode_steps=300,
)
