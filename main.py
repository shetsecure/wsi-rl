#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
from itertools import count

import sys
import yaml
import argparse
import datetime

from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gym_envs
import gymnasium as gym
from gymnasium.wrappers import TransformReward, TransformObservation

import utils
from models import CNN_LSTM, QValues
from agent import Agent
from strategy import EpsilonGreedyStrategy

device = torch.device("cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 7500 is the max mem_size for 32GB with patch_size=(128, 128), resize_thumbnail=(512, 512),
# 2500 is the max mem_size (28.5) for 32GB with patch_size=(128, 128), resize_thumbnail=(768, 768),

parser = argparse.ArgumentParser(
    description="Command-line parameters for Indexing experiments"
)

parser.add_argument(
    "-C",
    "--conf",
    type=str,
    required=True,
    dest="confpath",
    help="path of conf file",
)

c_time = datetime.datetime.now().strftime("%b_%d_%y %H:%M")


def create_env(config) -> gym.Env:
    gym_params = utils.GymParams(**config["gym"])

    env = gym.make(
        "gym_envs/WSIWorldEnv-v1",
        wsi_path="test.ndpi",
        render_mode=None,
        patch_size=gym_params.patch_size,
        resize_thumbnail=gym_params.resize_thumbnail,
        max_episode_steps=gym_params.max_episode_steps,
    )

    transform = transforms.Compose([transforms.ToTensor()])
    env = TransformReward(env, lambda r: torch.tensor([r]).to(device))
    env = TransformObservation(
        env,
        lambda obs: (
            transform(obs["current_view"]).unsqueeze(0).to(device),
            transform(obs["birdeye_view"]).unsqueeze(0).to(device),
            torch.tensor(obs["level"]).unsqueeze(0).to(device),
            torch.tensor(obs["p_coords"]).unsqueeze(0).to(device),
            torch.tensor(obs["b_rect"]).unsqueeze(0).to(device),
        ),
    )

    return env


def train(config, env: gym.Env):
    gym_params = utils.GymParams(**config["gym"])
    training_params = utils.TrainingParams(**config["train"])
    patch_size = env.unwrapped.wsi_wrapper.patch_size
    thumbnail_size = env.unwrapped.wsi_wrapper.thumbnail_size
    num_actions = env.action_space.n

    MODEL_PATH = f"DQN_b{training_params.batch_size}_m{training_params.memory_size}_pS{patch_size}_thS{thumbnail_size}_target_net_{c_time}.pt"
    writer = SummaryWriter(f"logs/{MODEL_PATH}")

    training_params_str = {
        f"training_params/{param_name}": param_value
        for param_name, param_value in training_params._asdict().items()
    }

    writer.add_hparams(training_params_str, {})
    writer.add_text("gym_params", str(gym_params))

    strategy = EpsilonGreedyStrategy(
        training_params.eps_start, training_params.eps_end, training_params.eps_decay
    )
    agent = Agent(strategy, num_actions, device)
    memory = utils.ReplayMemory(training_params.memory_size)

    policy_net = CNN_LSTM(patch_size, thumbnail_size, num_actions)
    target_net = CNN_LSTM(patch_size, thumbnail_size, num_actions)

    dummy_inputs = policy_net.get_dummy_inputs()
    writer.add_graph(policy_net, dummy_inputs)
    for i in dummy_inputs:
        del i

    policy_net, target_net = policy_net.to(device), target_net.to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=training_params.lr)

    for episode in range(training_params.num_episodes):
        observation, info = env.reset()
        loss_per_episode = 0
        reward_per_episode = 0

        for timestep in (pbar := tqdm(count(), total=gym_params.max_episode_steps)):
            pbar.set_description(f"Currently in Ep: {episode}")
            # current_view, birdeye_view, level, p_coords, b_rect

            action = agent.select_action(observation, policy_net)
            next_observation, reward, terminated, truncated, info = env.step(
                action.item()
            )
            reward_per_episode += reward

            patch, bird_view, level, p_coord, b_rect = observation
            (
                next_patch,
                next_bird_view,
                next_level,
                next_p_coord,
                next_b_rect,
            ) = next_observation

            memory.push(
                utils.RichExperience(
                    patch,
                    bird_view,
                    action,
                    level,
                    p_coord,
                    b_rect,
                    next_patch,
                    next_bird_view,
                    next_level,
                    next_p_coord,
                    next_b_rect,
                    reward,
                )
            )
            observation = next_observation

            if memory.can_provide_sample(training_params.batch_size):
                experiences = memory.sample(training_params.batch_size)
                (
                    patches,
                    bird_views,
                    levels,
                    p_coords,
                    b_rects,
                    actions,
                    rewards,
                    next_patches,
                    next_bird_views,
                    next_levels,
                    next_p_coords,
                    next_b_rects,
                ) = utils.extract_rich_experiences_tensors(experiences)
                current_q_values = QValues.get_current(
                    policy_net,
                    actions,
                    (
                        patches,
                        bird_views,
                        levels,
                        p_coords,
                        b_rects,
                    ),
                )
                next_q_values = QValues.get_next(
                    target_net,
                    (
                        next_patches,
                        next_bird_views,
                        next_levels,
                        next_p_coords,
                        next_b_rects,
                    ),
                )
                target_q_values = (next_q_values * training_params.gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                writer.add_scalar(f"Loss/ep{episode}_Timestep", loss, timestep)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_episode += loss.item()

            # TODO: Replace prints with loguru

            if terminated:
                print(f"Episode {episode} is terminated successfully")
                writer.add_scalar("SuccessfullEpisodes/Duration", timestep, episode)
                break

            if truncated:
                break

        writer.add_scalar("AllEpisodes/Duration", timestep, episode)
        writer.add_scalar("AllEpisodes/Reward", reward_per_episode, episode)

        writer.add_scalar("Loss/Episode", loss_per_episode, episode)

        writer.add_scalar(
            "GPU_MEM/Episode", round(torch.cuda.memory_allocated() / 1e9, 2), episode
        )

        if episode % training_params.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % training_params.saving_update == 0:
            print(f"Saving weights of the policy net")
            target_net.load_state_dict(policy_net.state_dict())
            MODEL_PATH = f"DQN_b{training_params.batch_size}_m{training_params.memory_size}_pS{patch_size}_thS{thumbnail_size}_target_net_{c_time}.pt"
            writer.add_text("Weights_path", MODEL_PATH, 0)
            torch.save(target_net.state_dict(), MODEL_PATH)

    env.close()
    writer.flush()


def main(argv):
    args = parser.parse_args(argv[1:])
    config = yaml.load(open(Path(args.confpath)), Loader=yaml.SafeLoader)

    env = create_env(config)

    train(config, env)

    env.close()


if __name__ == "__main__":
    main(sys.argv)
