#!/usr/bin/env python
# coding: utf-8


from __future__ import annotations

import random
from itertools import count
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import math

transform = transforms.Compose([transforms.ToTensor()])


# In[3]:


from collections import defaultdict
from tqdm import tqdm
import numpy as np
from collections import namedtuple

import gym_envs

import gymnasium as gym
from gym.wrappers import TransformReward, TransformObservation

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cpu")

env = gym.make(
    "gym_envs/WSIWorldEnv-v1",
    render_mode="rgb_array",
    patch_size=(64, 64),
    resize_thumbnail=(512, 512),
)
env = TransformReward(env, lambda r: torch.tensor([r]).to(device))
env = TransformObservation(
    env,
    lambda obs: (
        transform(obs[0]).unsqueeze(0).to(device),
        transform(obs[1]).unsqueeze(0).to(device),
    ),
)


class DQN(nn.Module):
    def __init__(self, patch_size, thumbnail_size, num_actions):
        super(DQN, self).__init__()
        self.patch_width, self.patch_height = patch_size
        self.thumbnail_width, self.thumbnail_height = thumbnail_size

        # Define the CNN layers for the current view
        self.current_view_conv = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.current_view_fc = nn.Linear(
            6 * self.patch_width * self.patch_height, 128
        )  # Update 256*256 based on the actual size of the feature maps

        # Define the CNN layers for the bird eye view
        self.birdeye_view_conv = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.birdeye_view_fc = nn.Linear(
            6 * self.thumbnail_width * self.thumbnail_height, 128
        )  # Update based on the actual size

        # # Attention mechanism
        # self.attention = nn.Linear(256, 256)

        # Fully connected layers after attention
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, current_view, birdeye_view):
        # Forward pass for current view
        x_current = F.relu(self.current_view_conv(current_view))
        x_current = x_current.view(
            -1, 6 * self.patch_width * self.patch_height
        )  # Update based on the actual size
        x_current = F.relu(self.current_view_fc(x_current))

        # Forward pass for bird eye view
        x_birdeye = F.relu(self.birdeye_view_conv(birdeye_view))
        x_birdeye = x_birdeye.view(
            -1, 6 * self.thumbnail_width * self.thumbnail_height
        )  # Update based on the actual size
        x_birdeye = F.relu(self.birdeye_view_fc(x_birdeye))

        # Concatenate the outputs of both branches
        x_combined = torch.cat((x_current, x_birdeye), dim=1)

        # # Attention mechanism
        # attention_weights = F.softmax(self.attention(x_combined), dim=1)
        # x_attention = torch.sum(attention_weights * x_combined, dim=0)

        # Fully connected layers after attention
        x = F.relu(self.fc1(x_combined))
        x = self.fc2(x)

        return x


patch_size = env.unwrapped.wsi_wrapper.patch_size
thumbnail_size = env.unwrapped.wsi_wrapper.thumbnail_size
num_actions = env.action_space.n  # Number of actions in your environment
# model = DQN(num_actions)
# sum(p.numel() for p in model.parameters())


# In[25]:


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, observation, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return (
                    policy_net(observation[0], observation[1])
                    .argmax()
                    .reshape(-1)
                    .to(self.device)
                )  # 7DI HNAYA .reshape(-1)


class QValues:
    @staticmethod
    def get_current(policy_net, patches, bird_views, actions):
        # gather to get the q value for the selected action that was judged as the best back then
        return policy_net(patches, bird_views).gather(
            dim=1, index=actions.unsqueeze(-1)
        )

    @staticmethod
    def get_next(target_net, next_patches, next_bird_views):
        values = target_net(next_patches, next_bird_views).max(dim=1)[0].detach()
        return values


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


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(
            -1.0 * current_step * self.decay
        )


# dqn = DQN(env.action_space.n).to(device)

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

    next_patches = torch.cat(batch.next_patches)  # torch.Size([144, 256, 256])
    next_bird_views = torch.cat(batch.next_bird_views)  # torch.Size([144, 828, 1650])

    return (patches, bird_views, actions, rewards, next_patches, next_bird_views)


batch_size = 48
gamma = 0.9
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 17000
lr = 0.001
num_episodes = 100000

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, env.action_space.n, device)
memory = ReplayMemory(memory_size)

policy_net = DQN(patch_size, thumbnail_size, env.action_space.n).to(device)
target_net = DQN(patch_size, thumbnail_size, env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []


for episode in range(num_episodes):
    observation, info = env.reset()

    for timestep in count():
        action = agent.select_action(observation, policy_net)
        next_observation, reward, terminated, truncated, info = env.step(action.item())

        patches, bird_views = observation
        next_patches, next_bird_views = next_observation

        memory.push(
            Experience(
                patches, bird_views, action, next_patches, next_bird_views, reward
            )
        )
        observation = next_observation

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            (
                patches,
                bird_views,
                actions,
                rewards,
                next_patches,
                next_bird_views,
            ) = extract_tensors(experiences)
            current_q_values = QValues.get_current(
                policy_net, patches, bird_views, actions
            )
            next_q_values = QValues.get_next(target_net, next_patches, next_bird_views)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if terminated:
            episode_durations.append(timestep)
            break

    print(episode_durations)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
# em_val.close()

MODEL_PATH = "target_net"
torch.save(target_net.state_dict(), MODEL_PATH)
