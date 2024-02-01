import torch
import random
from collections import namedtuple
from typing import NamedTuple, Union, Tuple, List


import gc
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import *

import matplotlib.pyplot as plt

from gym_envs.envs.wsi_env import WSIEnv


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)


class GymParams(NamedTuple):
    patch_size: Union[int, Tuple[int, int]]  # patch_size of the view of the agent
    resize_thumbnail: Union[int, Tuple[int, int]]  # size of the bird_view for the agent
    max_episode_steps: int


class TrainingParams(NamedTuple):
    batch_size: int
    gamma: float
    eps_start: int
    eps_end: float
    eps_decay: float
    target_update: int
    memory_size: int
    lr: float
    num_episodes: int
    saving_update: int


Experience = namedtuple(
    "Experience",
    ("patch", "bird_view", "action", "next_patch", "next_bird_view", "reward"),
)

RichExperience = namedtuple(
    "RichExperience",
    (
        "patch",
        "bird_view",
        "action",
        "level",
        "p_coord",
        "b_rect",
        "next_patch",
        "next_bird_view",
        "next_level",
        "next_p_coord",
        "next_b_rect",
        "reward",
    ),
)


def extract_tensors(experiences: List[Experience]):
    batch = Experience(*zip(*experiences))

    patches = torch.cat(batch.patches)  # torch.Size([B, 3, 256, 256])
    bird_views = torch.cat(batch.bird_views)  # torch.Size([B, 3, 828, 1650])

    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    next_patches = torch.cat(batch.next_patches)  # torch.Size([B, 3, 256, 256])
    next_bird_views = torch.cat(batch.next_bird_views)  # torch.Size([B, 3, 828, 1650])

    return (patches, bird_views, actions, rewards, next_patches, next_bird_views)


def extract_rich_experiences_tensors(experiences: List[RichExperience]):
    batch = RichExperience(*zip(*experiences))

    patches = torch.cat(batch.patch)
    bird_views = torch.cat(batch.bird_view)
    levels = torch.cat(batch.level)
    p_coords = torch.cat(batch.p_coord)
    b_rects = torch.cat(batch.b_rect)

    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    next_patches = torch.cat(batch.next_patch)
    next_bird_views = torch.cat(batch.next_bird_view)
    next_levels = torch.cat(batch.next_level)
    next_p_coords = torch.cat(batch.next_p_coord)
    next_b_rects = torch.cat(batch.next_b_rect)

    return (
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
    )


def get_gpu_mem_info() -> str:
    total_gb = round(torch.cuda.get_device_properties("cuda").total_memory / 1e9, 2)
    used_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

    return f"GPU MEM USED: {used_gb} / {total_gb}"


def simulate_run(env, training_params: TrainingParams, device="cuda"):
    """
    Simulate the worse case scneario to see if the exp will run or will end up OOM.
    Will just load dummy tensors and models and simulate it.

    If it pass, then probably it will continue training otherwise need to adjust the training_params
    """
    print(device)
    try:
        patch_size = env.unwrapped.wsi_wrapper.patch_size
        thumbnail_size = env.unwrapped.wsi_wrapper.thumbnail_size
        num_actions = env.action_space.n

        print(get_gpu_mem_info())

        batch_size = training_params.batch_size
        memory_size = training_params.memory_size

        # load the two models to memory
        print("Loading the models")
        policy_net = CNN_Attention(patch_size, thumbnail_size, num_actions).to(device)
        target_net = CNN_Attention(patch_size, thumbnail_size, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        print(get_gpu_mem_info())

        # load a batch to memory
        print(f"Loading dummy batch inputs with size {batch_size} ")
        batch = policy_net.get_dummy_inputs(batch_size, device=device)
        print(get_gpu_mem_info())

        # process it
        print("Flow of inputs into networks")
        out = policy_net(*batch)
        with torch.no_grad():
            out2 = target_net(*batch)

        print(get_gpu_mem_info())

        # load a dummy memory of experiences, batch_size * 2 cuz each xp have state_i and state_i+1
        memory = ReplayMemory(training_params.memory_size)

        print("Populating the memory bank with Experiences")

        for i in (pbar := tqdm(range(memory_size))):
            # load the actions and rewards that exists in memory
            pbar.set_description(
                f"Loading dummy exp number: {i} || GPU MEM USED: {get_gpu_mem_info()}"
            )
            action = torch.randint(low=0, high=num_actions, size=(1,)).to(device)
            reward = torch.randn(1).to(device)
            patch, bird_view, level, p_coord, b_rect = policy_net.get_dummy_inputs(
                1, device=device
            )
            (
                next_patch,
                next_bird_view,
                next_level,
                next_p_coord,
                next_b_rect,
            ) = policy_net.get_dummy_inputs(1, device=device)
            memory.push(
                RichExperience(
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

        # Loading optimizer and simulating a training loop
        print("Loading optimizer and simulating a training loop")
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(params=policy_net.parameters(), lr=training_params.lr)

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
        ) = extract_rich_experiences_tensors(experiences)
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

        optimizer.zero_grad()
        loss.backward()

        print("Emptying the GPU memory, deleting everything")
        for xp in memory.memory:
            del xp

        del (
            policy_net,
            target_net,
            batch,
            out,
            out2,
            memory,
            actions,
            rewards,
            patches,
            bird_views,
            levels,
            p_coords,
            b_rect,
            next_patches,
        )
        del next_bird_views, next_levels, next_p_coords, next_b_rect

        torch.cuda.empty_cache()
        gc.collect()

        print(get_gpu_mem_info())
        print(get_gpu_mem_info())

        return True
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        return False


def log_actions_histogram(
    writer, dictionary, step, tag, xticks=list(WSIEnv.action_to_name.values())
):
    x_values = list(dictionary.keys())
    y_values = list(dictionary.values())

    plt.bar(x_values, y_values)
    plt.xlabel("Actions")
    plt.ylabel("Count")
    plt.title(tag)

    plt.xticks(x_values, xticks, rotation=45)

    writer.add_figure(tag, plt.gcf(), global_step=step)
    plt.clf()
