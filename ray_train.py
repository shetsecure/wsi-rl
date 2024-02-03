import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gymnasium as gym

from gym_envs.envs.new_env import WSIWorldEnv
from ray.rllib.algorithms.ppo import PPOConfig

import ray
from ray import air, tune
from ray.rllib.algorithms.sac import SACConfig

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm import Algorithm


from ray.tune.registry import register_env


env_config = {
    "dataset_path": "/home/lipade/rl/dataset.csv",
    "resize_thumbnail": (640, 480),
    "patch_size": (84, 84),
    "train_mode": True,
    "max_episode_steps": 300,
}

register_env("myEnv", lambda config: WSIWorldEnv(config))


def train_ppo():
    rllib_config = (
        PPOConfig()
        .environment(env="myEnv", env_config=env_config)
        .resources(num_gpus=1, num_cpus_per_worker=6)
        .debugging(seed=0)
        .rollouts(num_rollout_workers=2)
        .training(
            lr=0.0000625,
            train_batch_size=128,
        )
        .evaluation(
            evaluation_interval=1000,
        )
    )

    stop_criteria = {
        "training_iteration": 1_000_000,
        "episode_reward_mean": -24,  # Or when the mean reward reaches 200
    }

    air_config = air.RunConfig(
        name="PPO",
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True, checkpoint_frequency=5000, num_to_keep=5
        ),
        log_to_file=True,
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=rllib_config,
        run_config=air_config,
    )

    ray.init()
    tuner.fit()
    ray.shutdown()


from ray.rllib.algorithms.algorithm import Algorithm


def restore_ppo(ckpt_path):
    env_config["render_mode"] = "human"

    config = (
        PPOConfig()
        .environment(env="myEnv", env_config=env_config)
        .resources(num_gpus=1, num_cpus_per_worker=6)
        .debugging(seed=0)
        .rollouts(num_rollout_workers=1)
    )
    algo = config.build()

    algo.restore(ckpt_path)

    env = algo.env_creator(env_config)
    env.metadata["render_fps"] = 1  # TODO: fix this
    # env = WSIWorldEnv(env_config)

    obs, info = env.reset()

    done = False
    truncated = False
    while not done and not truncated:
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(info)


if __name__ == "__main__":
    train_ppo()
    # restore_ppo(
    #     "/home/lipade/ray_results/PPO/PPO_myEnv_6bb3d_00000_0_2024-02-03_00-33-08/checkpoint_000001"
    # )
