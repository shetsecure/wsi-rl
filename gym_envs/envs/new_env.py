import numpy as np
import pandas as pd
import pyglet

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces

from . import helpers
from .wsi_wrapper import WSIApi


class WSIWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 5}
    action_to_name = {
        0: "Right",
        1: "Up",
        2: "Left",
        3: "Down",
        4: "Zoom-in",
        5: "Zoom-out",
        6: "Crop",
        # 7: "Stop",
    }

    def __init__(self, config: dict):
        # print("Dataset path ", dataset_path)
        self.__init_args(config)

        # np.random.seed(self.seed)

        # dataset path contains csv file that have the wsi_path and hdf5_path for the attention scores
        self.dataset_csv = pd.read_csv(self.dataset_path)

        # TODO: Add random selection here afterwards
        wsi_path, hdf5_path = self.dataset_csv.iloc[0]
        self._current_wsi = WSIApi(wsi_path, self.patch_size, self.resize_thumbnail)

        # Used in reward
        width, height = self._current_wsi.slide.dimensions
        self.max_possible_distance = np.sqrt(width**2 + height**2)

        # Defining the action and observation space
        self.action_space = spaces.Discrete(len(self.action_to_name))
        self.observation_space = spaces.Dict(
            {
                "current_view": spaces.Box(
                    low=0,
                    high=255,
                    shape=(*self.patch_size, 3),
                    dtype=np.uint8,
                ),
                "birdeye_view": spaces.Box(
                    low=0,
                    high=255,
                    # reverse it
                    shape=(self.resize_thumbnail[1], self.resize_thumbnail[0], 3),
                    dtype=np.uint8,
                ),
                # current level where the patch was taken from, binary encoded
                "level": spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8),
                # normalized coords of the patch with reference to the central point of the current WSI
                "p_coords": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                # normalized coords of the rectangle locating the patch with reference to the central point of the birdeye view
                "b_rect": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            }
        )

        self._action_to_behavior = {
            0: np.array([self.base_step_size, 0]),  # right
            1: np.array(
                [0, -self.base_step_size]
            ),  # up ( cuz of openslide it's flipped )
            2: np.array([-self.base_step_size, 0]),  # left
            3: np.array(
                [0, self.base_step_size]
            ),  # down ( cuz of openslide it's flipped )
            4: np.int8(-1),  # zoom in
            5: np.int8(1),  # zoom out
            6: True,  # Crop
            # 7: True,  # Stop
        }

        if self.train_mode:
            # Only in training mode we need the attention scores to calculate rewards.
            (
                self._predictive_coords,
                self._predictive_coords_scores,
            ) = helpers.get_most_predictive_patches_coords(hdf5_path)
            self._original_coords_scores = self._predictive_coords_scores.copy()
        else:
            self.predictive_coords = None
            self.predictive_coords_scores = None

        # Necessary to define these here, and a must to include them in reset()
        self.position = self._current_wsi.position
        self.patch_queue = helpers.PatchPriorityQueue(capacity=10)
        self.current_time = 0

    def reset(self, seed=None, options: dict = None):
        if not self.train_mode:
            if seed is None:
                seed = self.seed

            # np.random.seed(seed)

        super().reset(seed=seed, options=options)

        # We need to reset the attention scores and last visit time
        if self.train_mode:
            self._predictive_coords_scores = self._original_coords_scores.copy()

        # Reset the rest of the variables
        seed = self.seed if not self.train_mode else None
        self.position = self._current_wsi.reset(seed=seed, options=options)
        self.patch_queue = helpers.PatchPriorityQueue(capacity=10)
        self.current_time = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}. Valid actions are: {self.action_space}"
            )

        # Primarily used for the reward shaping
        self.current_time += 1

        if self.current_time >= self.max_episode_steps:
            observation, info = self._get_obs(), self._get_info()
            done = False
            truncated = True

            return observation, self._get_reward(action), done, truncated, info

        # # Update the attention scores
        # if self.train_mode:
        #     # Also part of the reward shaping
        #     self._update_attention_scores()

        # reward can be None if we are in testing mode
        reward = self._get_reward(action)

        if action == 7:
            # Stop action
            observation, info = self._get_obs(), self._get_info()
            done = True

            return observation, reward, done, False, info

        # TODO: Determine the patch priority score in testing mode
        elif action == 6:
            # Crop action
            p_score = reward
            self._crop_current_view(p_score)
            observation, info = self._get_obs(), self._get_info()

            return observation, reward, False, False, info

        # Update the view & position
        self._update_view(action)
        self.position = self._current_wsi.position

        observation, info = self._get_obs(), self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, False, False, info

    def _update_view(self, action: int):
        selected_action = self._action_to_behavior[action]

        if isinstance(selected_action, np.ndarray):
            dx, dy = (selected_action // self.base_step_size) * self._get_step_size()

            self._current_wsi.move(dx, dy)
        else:
            if selected_action == -1:
                self._current_wsi.zoom_in()
            elif selected_action == 1:
                self._current_wsi.zoom_out()

    # TODO: Figure out how to calculate the patch priority score in testing mode
    def _crop_current_view(self, patch_priority_score: float):
        """
        In training mode, the patch prioty score is the reward which is the attention score.
        In testing mode, the patch priority score is ??? (TODO)
        """
        patch_priority_score = (
            0.0 if patch_priority_score is None else patch_priority_score
        )
        self.patch_queue.add_patch_metadata(
            patch_priority_score,
            (self.position, self._current_wsi.level, self.patch_size),
        )

    def _get_step_size(self) -> int:
        # Calculate step size based on the current zoom level
        # Here, we exponentially decrease the step size with increasing zoom level
        # pixels at the lowest zoom level
        return int(
            self.base_step_size
            * np.exp(-self.step_decay_rate * self._current_wsi.level)
        )

    # TODO: Add the decay with last visit time + maybe the spatial decay as well
    def _get_reward(self, action: int = None, crop_base_reward=0.5):
        if action is not None:
            assert self.action_space.contains(action)

        if action == 7 and self.current_time < 20:
            # explore a lil bit before you end it
            return -5.0

        if self.train_mode:
            # Reward will be -distance from the current position to the nearest predictive patch
            assert self._predictive_coords is not None
            assert self._predictive_coords_scores is not None

            x, y = self.position
            dx, dy = self.patch_size

            # calculate the euclidean distance from the current position to the nearest predictive patch
            min_distance = np.inf
            # predictive_patch_index = None
            for i, (px, py) in enumerate(self._predictive_coords):
                distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    # predictive_patch_index = i

            reward = -min_distance / self.max_possible_distance

            """
            if action == 6:
                # Crop action, reward will be between -2.5 and 0.5, instead of -1 and 0.5
                # This can get the agent stuck to maximize the reward, we gonna get rid of it
                # Instead, we will track the most visited patches and gather them after the episode is finished.
                attention_score_multiplier = 2.0
                total_crop_reward = crop_base_reward + (
                    reward * attention_score_multiplier
                )
                return total_crop_reward
            """

            return reward
        else:
            return 1.0

    # TODO: Use wrappers instead to transform the observation
    def _get_obs(self):
        thumbnail_shape = np.asarray(self._current_wsi.thumbnail).shape[:2]
        bird_position = self._current_wsi.bird_position
        bird_view_size = self._current_wsi.bird_view_size

        return {
            "current_view": np.asarray(self._current_wsi.current_view, dtype=np.uint8),
            "birdeye_view": np.asarray(
                self._current_wsi.current_bird_view, dtype=np.uint8
            ),
            "level": np.asarray(
                helpers.binary_encoding(self._current_wsi.level, n_bits=4),
                dtype=np.uint8,
            ),
            "p_coords": np.asarray(
                helpers.map_coords_to_center_and_normalize(
                    self._current_wsi.slide.dimensions, *self._current_wsi.position
                ),
                dtype=np.float32,
            ),
            "b_rect": np.asarray(
                helpers.map_rect_info(thumbnail_shape, *bird_position, *bird_view_size),
                dtype=np.float32,
            ),
        }

    def _get_info(self):
        return {
            "coords": self._current_wsi.position,
            "zoom_level": self._current_wsi.level,
            "patch_size": self.patch_size,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        def pilImageToSurface(pilImage):
            return pyglet.image.ImageData(
                pilImage.size[0],
                pilImage.size[1],
                "RGB",
                pilImage.tobytes(),
                pitch=pilImage.size[0] * -3,
            ).get_texture()

        if self.window is None and self.clock is None:
            if self.render_mode == "human":
                self.window = pyglet.window.Window(*self.window_size)
                self.clock = pyglet.clock.Clock()

        canvas = pilImageToSurface(self._current_wsi.current_bird_view)

        if self.render_mode == "human":
            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()
            canvas.blit(0, 0)
            self.window.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(canvas.get_image_data().get_data()), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            self.window.close()

    def __init_args(self, config: dict):
        dataset_path = config["dataset_path"]
        patch_size = config["patch_size"] if "patch_size" in config else 128
        decay_lambda = config["decay_lambda"] if "decay_lambda" in config else 0.01
        decay_gamma = config["decay_gamma"] if "decay_gamma" in config else 0.001
        resize_thumbnail = (
            config["resize_thumbnail"] if "resize_thumbnail" in config else 512
        )
        render_mode = config["render_mode"] if "render_mode" in config else None
        train_mode = config["train_mode"] if "train_mode" in config else True
        seed = config["seed"] if "seed" in config else 24

        step_decay_rate = (
            config["step_decay_rate"] if "step_decay_rate" in config else 0.1
        )
        self.max_episode_steps = (
            config["max_episode_steps"] if "max_episode_steps" in config else 300
        )
        self.spec = EnvSpec("WSIWorldEnv-v1", max_episode_steps=self.max_episode_steps)

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Valid render modes are: {self.metadata['render_modes']}"
            )

        self.dataset_path = dataset_path
        self.decay_lambda = decay_lambda
        self.decay_gamma = decay_gamma
        self.step_decay_rate = step_decay_rate
        self.seed = seed

        self.patch_size = (
            (patch_size, patch_size)
            if not isinstance(patch_size, tuple)
            else patch_size
        )

        self.base_step_size = (
            config["base_step_size"]
            if "base_step_size" in config
            else self.patch_size[0]
        )

        self.resize_thumbnail = (
            (resize_thumbnail, resize_thumbnail)
            if not isinstance(resize_thumbnail, tuple)
            else resize_thumbnail
        )
        self.window_size = self.resize_thumbnail

        self.render_mode = render_mode
        self.train_mode = train_mode

        self.window = None
        self.clock = None
