import torch
import torch.nn as nn
import torch.nn.functional as F


class QValues:
    @staticmethod
    def get_current(policy_net, actions, args):
        # gather to get the q value for the selected action that was judged as the best back then
        return policy_net(*args).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, args):
        values = target_net(*args).max(dim=1)[0].detach()
        return values


class DualStreamCNN(nn.Module):
    def __init__(self, patch_size, thumbnail_size, n_actions):
        super(DualStreamCNN, self).__init__()
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

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_actions)

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

    def get_dummy_inputs(self, batch_size=1, device="cpu"):
        # Only used to construct the graph with tensorboard
        current_view = torch.randn(batch_size, 3, *self.patch_size).to(device)
        birdeye_view = torch.randn(batch_size, 3, *self.thumbnail_size).to(device)

        return current_view, birdeye_view


class CNN_LSTM(nn.Module):
    def __init__(self, patch_size=128, thumbnail_size=512, n_actions=6):
        super(CNN_LSTM, self).__init__()
        self.patch_size = (
            [patch_size] * 2 if isinstance(patch_size, int) else patch_size
        )
        self.thumbnail_size = (
            [thumbnail_size] * 2 if isinstance(thumbnail_size, int) else thumbnail_size
        )
        self.n_actions = n_actions

        self.patch_cnn = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.thumbnail_cnn = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for [level, p_coords, b_rect] total = 10
        self.additional_info_fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # calculating the combined feature size.
        dummy_patch, dummy_t = torch.randn(1, 3, *self.patch_size), torch.randn(
            1, 3, *self.thumbnail_size
        )
        p_embd, t_embd = self.patch_cnn(dummy_patch), self.thumbnail_cnn(dummy_t)
        p_embd_len, t_emd_len = (
            torch.prod(torch.tensor(p_embd.shape)).item(),
            torch.prod(torch.tensor(t_embd.shape)).item(),
        )
        del dummy_patch, dummy_t, p_embd, t_embd

        combined_features_size = (
            p_embd_len + t_emd_len + self.additional_info_fc[0].out_features,
        )

        # compression
        # TODO: explore how to force these two to learn the same representation
        self.latent_space = nn.Sequential(
            nn.Linear(combined_features_size[0], 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
        )

        self.latent_space_no_bn = nn.Sequential(
            nn.Linear(combined_features_size[0], 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
        )

        self.attention = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Softmax(dim=1)
        )

        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        # Fully connected layer for action effect prediction
        self.action_effect_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_actions),  # Output size depends
        )

    def forward(self, current_view, birdeye_view, level, p_coords, b_rect):
        # Process patch and thumbnail through their respective CNNs
        patch_features = self.patch_cnn(current_view)
        thumbnail_features = self.thumbnail_cnn(birdeye_view)

        # Flatten the outputs for concatenation
        patch_features = patch_features.view(patch_features.size(0), -1)
        thumbnail_features = thumbnail_features.view(thumbnail_features.size(0), -1)

        # Concatenate additional inputs after processing them through FC layers
        additional_info = torch.cat([level, p_coords, b_rect], dim=1)
        additional_info = self.additional_info_fc(additional_info)

        # Concatenate all features
        combined_features = torch.cat(
            [patch_features, thumbnail_features, additional_info], dim=1
        )

        # Compression
        if p_coords.shape[0] == 1:  # Batch size = 1
            latent_space_features = self.latent_space_no_bn(combined_features)
        else:
            latent_space_features = self.latent_space(combined_features)

        # Apply attention
        attention_weights = self.attention(latent_space_features)
        attended_features = latent_space_features * attention_weights

        # Process through LSTM
        lstm_out, _ = self.lstm(attended_features.unsqueeze(0))

        # Predict action effects
        action_effects = self.action_effect_fc(lstm_out.squeeze(0))

        return action_effects

    def get_dummy_inputs(self, batch_size=1, device="cpu", requires_grad=False):
        # Only used to construct the graph with tensorboard
        current_view = torch.randn(
            batch_size, 3, *self.patch_size, requires_grad=requires_grad
        ).to(device)
        birdeye_view = torch.randn(
            batch_size, 3, *self.thumbnail_size, requires_grad=requires_grad
        ).to(device)
        level = torch.tensor(
            [[0] * 4] * batch_size, dtype=torch.float32, requires_grad=requires_grad
        ).to(device)
        p_coords = torch.randn(batch_size, 2, requires_grad=requires_grad).to(device)
        b_rect = torch.randn(batch_size, 4, requires_grad=requires_grad).to(device)

        return current_view, birdeye_view, level, p_coords, b_rect


class CNN_Attention(nn.Module):
    def __init__(self, patch_size=128, thumbnail_size=512, n_actions=6):
        super(CNN_Attention, self).__init__()
        self.patch_size = (
            [patch_size] * 2 if isinstance(patch_size, int) else patch_size
        )
        self.thumbnail_size = (
            [thumbnail_size] * 2 if isinstance(thumbnail_size, int) else thumbnail_size
        )
        self.n_actions = n_actions

        self.patch_cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.thumbnail_cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for [level, p_coords, b_rect] total = 10
        self.additional_info_fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.3),
        )

        # calculating the combined feature size.
        dummy_patch, dummy_t = torch.randn(1, 3, *self.patch_size), torch.randn(
            1, 3, *self.thumbnail_size
        )
        p_embd, t_embd = self.patch_cnn(dummy_patch), self.thumbnail_cnn(dummy_t)
        p_embd_len, t_emd_len = (
            torch.prod(torch.tensor(p_embd.shape)).item(),
            torch.prod(torch.tensor(t_embd.shape)).item(),
        )
        del dummy_patch, dummy_t, p_embd, t_embd

        combined_features_size = (
            p_embd_len + t_emd_len + self.additional_info_fc[0].out_features,
        )

        # compression
        self.latent_space_no_bn = nn.Sequential(
            nn.Linear(combined_features_size[0], 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.attention = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Softmax(dim=1)
        )

        # Fully connected layer prediction
        self.action_fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, self.n_actions),  # Output size depends
        )

    def forward(self, current_view, birdeye_view, level, p_coords, b_rect):
        # Process patch and thumbnail through their respective CNNs
        patch_features = self.patch_cnn(current_view)
        thumbnail_features = self.thumbnail_cnn(birdeye_view)

        # Flatten the outputs for concatenation
        patch_features = patch_features.view(patch_features.size(0), -1)
        thumbnail_features = thumbnail_features.view(thumbnail_features.size(0), -1)

        # Concatenate additional inputs after processing them through FC layers
        additional_info = torch.cat([level, p_coords, b_rect], dim=1)
        additional_info = self.additional_info_fc(additional_info)

        # Concatenate all features
        combined_features = torch.cat(
            [patch_features, thumbnail_features, additional_info], dim=1
        )

        # Compression
        latent_space_features = self.latent_space_no_bn(combined_features)

        # Apply attention
        attention_weights = self.attention(latent_space_features)
        attended_features = latent_space_features * attention_weights

        # Predict action effects
        action_effects = self.action_fc(attended_features)

        return action_effects

    def get_dummy_inputs(self, batch_size=1, device="cpu", requires_grad=False):
        # Only used to construct the graph with tensorboard
        current_view = torch.randn(
            batch_size, 3, *self.patch_size, requires_grad=requires_grad
        ).to(device)
        birdeye_view = torch.randn(
            batch_size, 3, *self.thumbnail_size, requires_grad=requires_grad
        ).to(device)
        level = torch.tensor(
            [[0] * 4] * batch_size, dtype=torch.float32, requires_grad=requires_grad
        ).to(device)
        p_coords = torch.randn(batch_size, 2, requires_grad=requires_grad).to(device)
        b_rect = torch.randn(batch_size, 4, requires_grad=requires_grad).to(device)

        return current_view, birdeye_view, level, p_coords, b_rect


class GRU_CNN_Attention(nn.Module):
    def __init__(self, patch_size=128, thumbnail_size=512, n_actions=6):
        super(GRU_CNN_Attention, self).__init__()
        self.patch_size = (
            [patch_size] * 2 if isinstance(patch_size, int) else patch_size
        )
        self.thumbnail_size = (
            [thumbnail_size] * 2 if isinstance(thumbnail_size, int) else thumbnail_size
        )
        self.n_actions = n_actions

        self.patch_cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.thumbnail_cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        # calculating the combined feature size.
        dummy_patch, dummy_t = torch.randn(1, 3, *self.patch_size), torch.randn(
            1, 3, *self.thumbnail_size
        )
        p_embd, t_embd = self.patch_cnn(dummy_patch), self.thumbnail_cnn(dummy_t)
        p_embd_len, t_emd_len = (
            torch.prod(torch.tensor(p_embd.shape)).item(),
            torch.prod(torch.tensor(t_embd.shape)).item(),
        )
        del dummy_patch, dummy_t, p_embd, t_embd

        combined_features_size = p_embd_len + t_emd_len + 10

        # compression
        self.latent_space = nn.Sequential(
            nn.Linear(combined_features_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 128), nn.Softmax(dim=1)
        )

        self.gru = nn.GRU(
            input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5
        )

        # Fully connected layer prediction
        self.action_fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_actions),  # Output size depends
        )

    def forward(self, current_view, birdeye_view, level, p_coords, b_rect):
        # Process patch and thumbnail through their respective CNNs
        patch_features = self.patch_cnn(current_view)
        thumbnail_features = self.thumbnail_cnn(birdeye_view)

        # Flatten the outputs for concatenation
        patch_features = patch_features.view(patch_features.size(0), -1)
        thumbnail_features = thumbnail_features.view(thumbnail_features.size(0), -1)

        # Concatenate all features
        combined_features = torch.cat(
            [patch_features, thumbnail_features, level, p_coords, b_rect], dim=1
        )

        # Compression
        latent_space_features = self.latent_space(combined_features)

        # Apply attention
        attention_weights = self.attention(latent_space_features)
        attended_features = latent_space_features * attention_weights

        # Gru
        gru_out, _ = self.gru(attended_features)

        # Predict action effects
        action_effects = self.action_fc(gru_out)

        return action_effects

    def get_dummy_inputs(self, batch_size=1, device="cpu", requires_grad=False):
        # Only used to construct the graph with tensorboard
        current_view = torch.randn(
            batch_size, 3, *self.patch_size, requires_grad=requires_grad
        ).to(device)
        birdeye_view = torch.randn(
            batch_size, 3, *self.thumbnail_size, requires_grad=requires_grad
        ).to(device)
        level = torch.tensor(
            [[0] * 4] * batch_size, dtype=torch.float32, requires_grad=requires_grad
        ).to(device)
        p_coords = torch.randn(batch_size, 2, requires_grad=requires_grad).to(device)
        b_rect = torch.randn(batch_size, 4, requires_grad=requires_grad).to(device)

        return current_view, birdeye_view, level, p_coords, b_rect
