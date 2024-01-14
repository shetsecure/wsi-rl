import torch
import random
from collections import namedtuple
from typing import NamedTuple, Union, Tuple, List


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
