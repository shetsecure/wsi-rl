#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
from itertools import count

import os
import gc
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
from models import *
from agent import Agent
from strategy import EpsilonGreedyStrategy
from debug_tools import plot_grad_flow

try:
    if "lipade" in os.getlogin():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
except OSError:
    device = torch.device("cuda")

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
            torch.tensor(obs["p_coords"] * 100).unsqueeze(0).to(device),
            torch.tensor(obs["b_rect"] * 100).unsqueeze(0).to(device),
        ),
    )

    return env


def train(config, env: gym.Env):
    gym_params = utils.GymParams(**config["gym"])
    training_params = utils.TrainingParams(**config["train"])
    patch_size = env.unwrapped.wsi_wrapper.patch_size
    thumbnail_size = env.unwrapped.wsi_wrapper.thumbnail_size
    num_actions = env.action_space.n

    policy_net = GRU_CNN_Attention(patch_size, thumbnail_size, num_actions)
    target_net = GRU_CNN_Attention(patch_size, thumbnail_size, num_actions)

    model_name = str(policy_net.__class__).split(".")[-1].split("'")[0]
    MODEL_PATH = f"{model_name}_b{training_params.batch_size}_m{training_params.memory_size}_pS{patch_size}_thS{thumbnail_size}_target_net_{c_time}.pt"

    writer = SummaryWriter(f"logs_{model_name}/{MODEL_PATH}")

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

    dummy_inputs = policy_net.get_dummy_inputs()
    writer.add_graph(policy_net, dummy_inputs)
    for i in dummy_inputs:
        del i
    torch.cuda.empty_cache()
    gc.collect()

    policy_net, target_net = policy_net.to(device), target_net.to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=training_params.lr)

    for episode in range(training_params.num_episodes):
        observation, info = env.reset()
        loss_per_episode = 0
        reward_per_episode = 0

        saved_grad_episode = False
        actions_count = {a: c for a, c in zip(range(6), [0] * 6)}

        for timestep in (pbar := tqdm(count(), total=gym_params.max_episode_steps)):
            pbar.set_description(
                f"Currently in Ep: {episode} || {utils.get_gpu_mem_info()}"
            )
            # current_view, birdeye_view, level, p_coords, b_rect

            action = agent.select_action(observation, policy_net)
            actions_count[int(action)] += 1

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
                    patch=patch,
                    bird_view=bird_view,
                    action=action,
                    level=level,
                    p_coord=p_coord,
                    b_rect=b_rect,
                    next_patch=next_patch,
                    next_bird_view=next_bird_view,
                    next_level=next_level,
                    next_p_coord=next_p_coord,
                    next_b_rect=next_b_rect,
                    reward=reward,
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
                # writer.add_scalar(f"Loss/ep{episode}_Timestep", loss, timestep)

                optimizer.zero_grad()
                loss.backward()
                # plot_grad_flow(policy_net.named_parameters())

                if episode % 100 == 0 and not saved_grad_episode:
                    saved_grad_episode = True
                    print("Saving grads")
                    for name, param in policy_net.named_parameters():
                        if param.requires_grad and ("bias" not in name):
                            try:
                                writer.add_histogram(
                                    f"{name}.grad",
                                    param.grad.data.cpu().numpy(),
                                    global_step=episode,
                                )
                            except AttributeError:
                                # This happenes cuz of the compression layer since only one of the two is used
                                continue

                optimizer.step()

                loss_per_episode += loss.item()

            # TODO: Replace prints with loguru

            if terminated or truncated:
                break

        writer.add_scalar("AllEpisodes/Duration", timestep, episode)
        writer.add_scalar("AllEpisodes/Reward", reward_per_episode, episode)

        writer.add_scalar("Loss/Episode", loss_per_episode, episode)
        print(actions_count)
        utils.log_actions_histogram(
            writer, actions_count, episode, "actions_count_histogram"
        )

        if episode % training_params.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % training_params.saving_update == 0:
            print(f"Saving weights of the policy net")
            target_net.load_state_dict(policy_net.state_dict())
            writer.add_text("Weights_path", MODEL_PATH, 0)
            torch.save(target_net.state_dict(), MODEL_PATH)

    env.close()
    writer.flush()


def main(argv):
    args = parser.parse_args(argv[1:])
    config = yaml.load(open(Path(args.confpath)), Loader=yaml.SafeLoader)

    env = create_env(config)

    # simulate first
    training_params = utils.TrainingParams(**config["train"])
    training_params_str = {
        f"training_params/{param_name}": param_value
        for param_name, param_value in training_params._asdict().items()
    }
    print(f"Testing if the following config can fit the memory first")
    print(training_params_str)

    # TODO: Fix bug here: We're overestimating too much.
    # can_run_xp = utils.simulate_run(env, training_params, device)

    # if can_run_xp:
    #     train(config, env)
    # else:
    #     print("Can't run xp. Can't fit the whole thing into memory")

    train(config, env)

    env.close()


if __name__ == "__main__":
    main(sys.argv)
