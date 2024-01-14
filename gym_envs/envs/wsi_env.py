import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from . import helpers
from .wsi_wrapper import WSIApi


def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode
    ).convert()


class WSIWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self, wsi_path, patch_size=(256, 256), resize_thumbnail=False, render_mode=None
    ):
        self.wsi_wrapper = WSIApi(
            wsi_path, patch_size=patch_size, resize_thumbnail=resize_thumbnail
        )
        self.window_size = (
            resize_thumbnail
            if not isinstance(resize_thumbnail, bool)
            else self.wsi_wrapper.thumbnail_size
        )
        self.size = self.wsi_wrapper.slide.dimensions

        # We have 6 actions, corresponding to "right", "up", "left", "down", zoom-in, zoom-out
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict(
            {
                "current_view": spaces.Box(
                    low=0, high=255, dtype=np.uint8
                ),  # Variable-sized patch
                "birdeye_view": spaces.Box(
                    low=0, high=255, dtype=np.uint8
                ),  # Variable-sized thumbnail
                "level": spaces.Box(
                    low=0,
                    high=1,
                    shape=(4,),  # to represent all the possible levels < 10
                    dtype=np.uint8,
                ),  # current level where the patch was taken from, binary encoded
                "p_coords": spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),  # normalized coords of the patch with reference to the central point of the current WSI
                "b_rect": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(4,),  # (x,y, %width, %height)
                    dtype=np.float32,
                ),  # normalized coords of the rectangle locating the patch with reference to the central point of the birdeye view
            }
        )

        # self.observation_space = spaces.Tuple(
        #     [
        #         spaces.Box(low=0, high=255, dtype=np.uint8),  # Variable-sized patch
        #         spaces.Box(low=0, high=255, dtype=np.uint8),  # Variable-sized thumbnail
        #     ]
        # )

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([100, 0]),  # right
            1: np.array([0, -100]),  # up ( cuz of openslide it's flipped )
            2: np.array([-100, 0]),  # left
            3: np.array([0, 100]),  # down ( cuz of openslide it's flipped )
            4: np.int8(-1),  # zoom in
            5: np.int8(1),  # zoom out
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

    def step(self, discrete_action):
        # Map the action (element of {0,1,2,3,4,5}) to what we gonna do
        direction = self._action_to_direction[discrete_action]
        current_zoom_level = self.wsi_wrapper.level
        max_allowed_level = self.wsi_wrapper.max_allowed_level

        if isinstance(direction, np.ndarray):
            self.wsi_wrapper.move(*direction)
        else:
            if direction == -1:
                self.wsi_wrapper.zoom_in()
            else:
                self.wsi_wrapper.zoom_out()

        # An episode is done iff the agent has reached a view with a tissue_percentage < 0.1
        current_tissue_percentage = helpers.get_tissue_percentage(
            self.wsi_wrapper.current_view
        )
        terminated = current_tissue_percentage < 0.1

        # reward = -1 if not terminated else 0  # Binary sparse rewards
        # We need to minimazie the tissue_percenentage
        reward = -current_tissue_percentage  # dense reward

        # punish the agent if it tries to zoom-in when it's in level 0 or zoom-out when it's in the max_allowed_level.
        # TODO: either fix max_allowed_level for all slides or make it also a part of the observation
        if (current_zoom_level == 0 and discrete_action == -1) or (
            current_zoom_level == max_allowed_level and discrete_action == 1
        ):
            reward = -2.0

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
        # return (
        #     np.asarray(self.wsi_wrapper.current_view, dtype=np.uint8),
        #     np.asarray(self.wsi_wrapper.current_bird_view, dtype=np.uint8),
        # )

    def _get_info(self):
        return {
            "percentage_tissue": helpers.get_tissue_percentage(
                self.wsi_wrapper.current_view
            ),
            "coords": self.wsi_wrapper.position,
            "zoom_level": self.wsi_wrapper.level,
            "patch_size": self.wsi_wrapper.patch_size,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pilImageToSurface(self.wsi_wrapper.current_bird_view)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
