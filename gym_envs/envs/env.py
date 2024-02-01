import numpy as np
import pandas as pd
import pyglet

import gymnasium as gym
from gymnasium import spaces

from . import helpers
from .wsi_wrapper import WSIApi


class WSIWorldEnv(gym.Env):
    """
    Gym environment for Whole Slide Image (WSI) world.

    This environment simulates the navigation and exploration of a Whole Slide Image (WSI) dataset.
    The agent can take actions to move in different directions, zoom in/out, crop the current view, or stop the exploration.
    The environment provides observations of the current view, bird's eye view, level, patch coordinates, and rectangle information.
    The agent receives rewards based on the attention scores of the current view.

    Args:
        dataset_path (str): Path to the dataset containing CSV file with WSI and HDF5 paths.
        patch_size (tuple, optional): Size of the patch in pixels. Defaults to (128, 128).
        decay_lambda (float, optional): Decay factor for spatial decay component. Defaults to 0.01.
        decay_gamma (float, optional): Decay factor for time-based decay component. Defaults to 0.001.
        resize_thumbnail (tuple, optional): Size of the thumbnail image. Defaults to (512, 512).
        render_mode (str, optional): Rendering mode. Can be "human", "rgb_array", or None. Defaults to None.
        train_mode (bool, optional): Training mode flag. If True, attention scores are used for rewards. Defaults to True.
        seed (int, optional): Random seed. Defaults to 43.

    Attributes:
        metadata (dict): Metadata for the gym environment.
        action_to_name (dict): Mapping of action index to action name.

    """

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

    def __init__(
        self,
        dataset_path,
        patch_size=(128, 128),
        decay_lambda=0.01,
        decay_gamma=0.001,
        resize_thumbnail=(512, 512),
        render_mode=None,
        train_mode=True,
        seed=43,
        base_step_size=100,
        step_decay_rate=0.1,
    ):
        self.__init_args(
            dataset_path,
            patch_size,
            decay_lambda,
            decay_gamma,
            resize_thumbnail,
            render_mode,
            train_mode,
            seed,
            base_step_size,
            step_decay_rate,
        )

        np.random.seed(self.seed)

        # dataset path contains csv file that have the wsi_path and hdf5_path for the attention scores
        self.dataset_csv = pd.read_csv(self.dataset_path)

        # TODO: Add random selection here afterwards
        wsi_path, hdf5_path = self.dataset_csv.iloc[0]
        self._current_wsi = WSIApi(wsi_path, self.patch_size, self.resize_thumbnail)

        # Defining the action and observation space
        self.action_space = spaces.Discrete(len(self.action_to_name))
        self.observation_space = spaces.Dict(
            {
                "current_view": spaces.Box(
                    low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
                ),
                "birdeye_view": spaces.Box(
                    low=0, high=255, shape=(512, 512, 3), dtype=np.uint8
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
            0: np.array([100, 0]),  # right
            1: np.array([0, -100]),  # up ( cuz of openslide it's flipped )
            2: np.array([-100, 0]),  # left
            3: np.array([0, 100]),  # down ( cuz of openslide it's flipped )
            4: np.int8(-1),  # zoom in
            5: np.int8(1),  # zoom out
            6: True,  # Crop
            # 7: True,  # Stop
        }

        if self.train_mode:
            # Only in training mode we need the attention scores to calculate rewards.
            self._attention_scores = helpers.create_attention_score_map_l0(
                wsi_path, hdf5_path, self.patch_size, normalize=True
            )
            self._original_attention_scores = self._attention_scores.copy()
            self._last_visit_time = np.zeros(
                self._attention_scores.shape, dtype=np.uint8
            )
        else:
            self._attention_scores = None
            self._last_visit_time = None

        # Necessary to define these here, and a must to include them in reset()
        self.position = self._current_wsi.position
        self.patch_queue = helpers.PatchPriorityQueue(capacity=10)
        self.current_time = 0

        if self.render_mode == "human":
            self.window = None
            self.clock = None

    def reset(self, seed=None, options: dict = None):
        if not self.train_mode:
            if seed is None:
                seed = self.seed

            np.random.seed(seed)

        super().reset(seed=seed, options=options)

        # We need to reset the attention scores and last visit time
        if self.train_mode:
            self._attention_scores = self._original_attention_scores.copy()
            self._last_visit_time = np.zeros(
                self._attention_scores.shape, dtype=np.uint8
            )

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

        # Update the attention scores
        if self.train_mode:
            # Also part of the reward shaping
            self._update_attention_scores()

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

    def _get_reward(self, action: int = None, crop_base_reward=0.5):
        if action is not None:
            assert self.action_space.contains(action)

        if action == 7 and self.current_time < 20:
            # explore a lil bit before you end it
            return -5.0

        if self.train_mode:
            # Reward will be the mean of the attention scores of the current view
            assert self._attention_scores is not None

            x, y = self.position
            dx, dy = self.patch_size

            mean_att_score = self._attention_scores[
                y : y + dy * 2**self._current_wsi.level,
                x : x + dx * 2**self._current_wsi.level,
            ].mean()

            if action == 6:
                # Crop action, reward will be between -2.5 and 2.5, instead of -1 and 1
                attention_score_multiplier = 2.0
                total_crop_reward = crop_base_reward + (
                    mean_att_score * attention_score_multiplier
                )
                return total_crop_reward

            return mean_att_score

    # TODO: Add the spacial decay component & See if we can make it faster & momory efficient
    def _update_attention_scores(self):
        y, x = self.position
        dy, dx = self.patch_size

        update_area = (
            slice(y, y + dy * 2**self._current_wsi.level),
            slice(x, x + dx * 2**self._current_wsi.level),
        )
        time_elapsed = self.current_time - self._last_visit_time[update_area]
        # e−λ×D(i,j,x,y) [Spatial Decay]
        # self.attention_scores[update_area] *= np.exp(-self.decay_lambda * time_elapsed)
        # −γ×ΔT [Time-Based Decay Component]
        # self._attention_scores[update_area] -= self.decay_gamma * time_elapsed

        # Trying to avoid more mem alloc
        np.subtract(
            self._attention_scores[update_area],
            self.decay_gamma * time_elapsed,
            out=self._attention_scores[update_area],
        )

        np.clip(
            self._attention_scores[update_area],
            -1,
            1,
            out=self._attention_scores[update_area],
        )

        # Update last update time for the area
        self._last_visit_time[update_area] = self.current_time

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

    def __init_args(
        self,
        dataset_path,
        patch_size,
        decay_lambda,
        decay_gamma,
        resize_thumbnail,
        render_mode,
        train_mode,
        seed,
        base_step_size,
        step_decay_rate,
    ):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Valid render modes are: {self.metadata['render_modes']}"
            )

        self.dataset_path = dataset_path
        self.decay_lambda = decay_lambda
        self.decay_gamma = decay_gamma
        self.base_step_size = base_step_size
        self.step_decay_rate = step_decay_rate
        self.seed = seed

        self.patch_size = (
            (patch_size, patch_size)
            if not isinstance(patch_size, tuple)
            else patch_size
        )
        self.resize_thumbnail = (
            (resize_thumbnail, resize_thumbnail)
            if not isinstance(resize_thumbnail, tuple)
            else resize_thumbnail
        )
        self.window_size = self.resize_thumbnail

        self.render_mode = render_mode
        self.train_mode = train_mode
