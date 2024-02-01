import gymnasium as gym
import gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

env_kwargs = {
    "patch_size": 128,
    "resize_thumbnail": 512,
    "max_episode_steps": 300,
    "dataset_path": "dataset.csv",
    "base_step_size": 128,
}

env = make_vec_env("WSIWorldEnv-v1", n_envs=1, env_kwargs=env_kwargs)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=2,
    gamma=0.85,
    tensorboard_log="../ppo/",
    ent_coef=0.01,  # https://youtu.be/1ppslywmIPs?t=342
)

checkpoint_callback = CheckpointCallback(
    save_freq=1e3, save_path="./ppo_wsiworld_checkpoints/"
)

model.learn(
    total_timesteps=1_000_000, callback=[checkpoint_callback], progress_bar=True
)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model.save("ppo_wsiworld")


# model = PPO.load("ppo_wsiworld")

# env_kwargs = {
#     "patch_size": 128,
#     "resize_thumbnail": (512, 512),
#     "max_episode_steps": 100,
#     "render_mode": "human",
#     "dataset_path": "dataset.csv",
# }

# env = make_vec_env(
#     "WSIWorldEnv-v1", n_envs=1, env_kwargs=env_kwargs
# )  # n_envs=1 for single environment
# obs = env.reset()

# for i in range(1000):  # Set this to the desired number of steps
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()  # If your environment has a render method
#     if dones:
#         obs = env.reset()

# env.close()
