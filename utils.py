import torch
import random
from collections import namedtuple
from typing import NamedTuple, Union, Tuple


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
    ("patches", "bird_views", "action", "next_patches", "next_bird_views", "reward"),
)


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    patches = torch.cat(batch.patches)  # torch.Size([B, 3, 256, 256])
    bird_views = torch.cat(batch.bird_views)  # torch.Size([B, 3, 828, 1650])

    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    next_patches = torch.cat(batch.next_patches)  # torch.Size([B, 3, 256, 256])
    next_bird_views = torch.cat(batch.next_bird_views)  # torch.Size([B, 3, 828, 1650])

    return (patches, bird_views, actions, rewards, next_patches, next_bird_views)
