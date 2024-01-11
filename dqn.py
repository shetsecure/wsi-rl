import torch
import torch.nn as nn
import torch.nn.functional as F


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
