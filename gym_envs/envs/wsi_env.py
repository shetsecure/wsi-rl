import numpy as np
import pyglet

import gymnasium as gym
from gymnasium import spaces

from . import helpers
from .wsi_wrapper import WSIApi

from tqdm import trange


def pilImageToSurface(pilImage):
    return pyglet.image.ImageData(
        pilImage.size[0],
        pilImage.size[1],
        "RGB",
        pilImage.tobytes(),
        pitch=pilImage.size[0] * -3,
    ).get_texture()


class WSIEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}
    action_to_name = {
        0: "Right",
        1: "Up",
        2: "Left",
        3: "Down",
        4: "Zoom-in",
        5: "Zoom-out",
        6: "Crop",
    }

    def __init__(
        self,
        wsi_path,
        hdf5_path=None,
        patch_size=(256, 256),
        resize_thumbnail=False,
        render_mode=None,
        decay_lambda=0.01,
        decay_gamma=0.001,
    ):
        self.wsi_wrapper = WSIApi(
            wsi_path, patch_size=patch_size, resize_thumbnail=resize_thumbnail
        )
        self.window_size = (
            resize_thumbnail
            if not isinstance(resize_thumbnail, bool)
            else self.wsi_wrapper.thumbnail_size
        )
        self.window_size = (
            (resize_thumbnail, resize_thumbnail)
            if not isinstance(resize_thumbnail, tuple)
            else resize_thumbnail
        )
        self.patch_size = (
            (patch_size, patch_size)
            if not isinstance(patch_size, tuple)
            else patch_size
        )

        self.size = self.wsi_wrapper.slide.dimensions
        self.current_time = 0
        self.position = self.wsi_wrapper.position
        self.patch_queue = helpers.PatchPriorityQueue(capacity=10)

        if hdf5_path is not None:
            self.attention_scores = helpers.create_attention_score_map_l0(
                wsi_path, hdf5_path, self.patch_size, normalize=True
            )
            # self.attention_scores = helpers.normalize_scores(self.attention_scores)

            self.last_visit_time = np.zeros(self.attention_scores.shape, dtype=np.uint8)

            self.decay_lambda = decay_lambda
            self.decay_gamma = decay_gamma
        else:
            self.attention_scores = None
            self.last_visit_time = None
        # We have 6 actions, corresponding to "right", "up", "left", "down", zoom-in, zoom-out
        self.action_space = spaces.Discrete(6)

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

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_space_mapping = {
            0: np.array([100, 0]),  # right
            1: np.array([0, -100]),  # up ( cuz of openslide it's flipped )
            2: np.array([-100, 0]),  # left
            3: np.array([0, 100]),  # down ( cuz of openslide it's flipped )
            4: np.int8(-1),  # zoom in
            5: np.int8(1),  # zoom out
            6: True,  # Crop
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.done = False
        super().reset(seed=seed)
        self.wsi_wrapper.reset(seed, options)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def update_attention_scores(self):
        y, x = self.position
        dy, dx = self.patch_size

        self.current_time += 1

        update_area = (
            slice(y, y + dy * 2**self.wsi_wrapper.level),
            slice(x, x + dx * 2**self.wsi_wrapper.level),
        )
        time_elapsed = self.current_time - self.last_visit_time[update_area]
        # e−λ×D(i,j,x,y) [Spatial Decay]
        # self.attention_scores[update_area] *= np.exp(-self.decay_lambda * time_elapsed)
        # −γ×ΔT [Time-Based Decay Component]
        self.attention_scores[update_area] -= self.decay_gamma * time_elapsed

        # Optionally, clip values within a range, if needed
        np.clip(
            self.attention_scores[update_area],
            -1,
            1,
            out=self.attention_scores[update_area],
        )

        # Update last update time for the area
        self.last_visit_time[update_area] = self.current_time

        # ------------------------------------------------------------------------------------
        # Adjust last visit time for current patch
        # chunk_size = 10000  # Define a suitable chunk size

        # # Process in chunks
        # for i in trange(
        #     0,
        #     self.attention_scores.shape[0],
        #     chunk_size,
        #     desc="Updating the attention scores",
        # ):
        #     end = min(i + chunk_size, self.attention_scores.shape[0])
        #     self.attention_scores[i:end] *= (
        #
        #         np.exp(-self.decay_lambda * time_elapsed[i:end])
        #
        #         - self.decay_gamma * time_elapsed[i:end]
        #     )
        #     np.clip(
        #         self.attention_scores[i:end], -1, 1, out=self.attention_scores[i:end]
        #     )

        # self.attention_scores *= np.exp(-self.decay_lambda * time_elapsed)

        # self.attention_scores -= self.decay_gamma * time_elapsed

        # self.attention_scores = np.clip(self.attention_scores, -1, 1)

    def calculate_reward(self):
        # Reward will be the mean of the attention scores of the current view
        assert self.attention_scores is not None

        x, y = self.position
        dx, dy = self.patch_size

        mean_att_score = self.attention_scores[
            y : y + dy * 2**self.wsi_wrapper.level,
            x : x + dx * 2**self.wsi_wrapper.level,
        ].mean()

        return mean_att_score

    def get_step_size(self):
        # Calculate step size based on the current zoom level
        # Here, we exponentially decrease the step size with increasing zoom level
        base_step_size = 100  # pixels at the lowest zoom level
        return int(base_step_size * np.exp(-0.5 * self.wsi_wrapper.level))

    def crop_current_view(self, attention_score):
        self.patch_queue.add_patch_metadata(
            attention_score, (self.position, self.wsi_wrapper.level, self.patch_size)
        )

    def step(self, discrete_action):
        selected_action = self.action_space_mapping[discrete_action]

        # print(self.action_to_name[int(discrete_action)])
        # current_zoom_level = self.wsi_wrapper.level
        # max_allowed_level = self.wsi_wrapper.max_allowed_level

        crop_current_patch = False

        if isinstance(selected_action, np.ndarray):
            dx = dy = self.get_step_size()
            self.wsi_wrapper.move(dx, dy)
        else:
            if selected_action == -1:
                self.wsi_wrapper.zoom_in()
            elif selected_action == 1:
                self.wsi_wrapper.zoom_out()
            else:
                crop_current_patch = True

        self.position = self.wsi_wrapper.position

        # An episode is done iff the agent has reached a view with a tissue_percentage < 0.1
        # current_tissue_percentage = helpers.get_tissue_percentage(
        #     self.wsi_wrapper.current_view
        # )
        # terminated = current_tissue_percentage < 0.1
        self.update_attention_scores()

        # print("Calculating reward")
        self.reward = reward = self.calculate_reward()

        if crop_current_patch:
            self.crop_current_view(
                reward,
            )

        # Let's see what happens
        terminated = False

        # if terminated:
        #     reward = 10  # terminal reward to reinforce the objective
        # else:
        #     # reward = -1 if not terminated else 0  # Binary rewards
        #     # We need to minimazie the tissue_percenentage
        #     reward = -current_tissue_percentage  # dense reward

        #     # punish the agent if it tries to zoom-in when it's in level 0 or zoom-out when it's in the max_allowed_level.
        #     # TODO: either fix max_allowed_level for all slides or make it also a part of the observation
        #     if (current_zoom_level == 0 and discrete_action == -1) or (
        #         current_zoom_level == max_allowed_level and discrete_action == 1
        #     ):
        #         reward = -2.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        thumbnail_shape = np.asarray(self.wsi_wrapper.thumbnail).shape[:2]
        bird_position = self.wsi_wrapper.bird_position
        bird_view_size = self.wsi_wrapper.bird_view_size

        return {
            "current_view": np.asarray(self.wsi_wrapper.current_view, dtype=np.uint8),
            "birdeye_view": np.asarray(
                self.wsi_wrapper.current_bird_view, dtype=np.uint8
            ),
            "level": np.asarray(
                helpers.binary_encoding(self.wsi_wrapper.level, n_bits=4),
                dtype=np.uint8,
            ),
            "p_coords": np.asarray(
                helpers.map_coords_to_center_and_normalize(
                    self.wsi_wrapper.slide.dimensions, *self.wsi_wrapper.position
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
            "coords": self.wsi_wrapper.position,
            "zoom_level": self.wsi_wrapper.level,
            "patch_size": self.patch_size,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.clock is None:
            if self.render_mode == "human":
                self.window = pyglet.window.Window(*self.window_size)
                self.clock = pyglet.clock.Clock()

        canvas = pilImageToSurface(self.wsi_wrapper.current_bird_view)

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
