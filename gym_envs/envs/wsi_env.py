import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from . import helpers
from .wsi_wrapper import WSIWrapper


def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode
    ).convert()


class WSIWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self, wsi_path, patch_size=(256, 256), resize_thumbnail=False, render_mode=None
    ):
        self.wsi_wrapper = WSIWrapper(
            wsi_path, patch_size=patch_size, resize_thumbnail=resize_thumbnail
        )
        self.window_size = resize_thumbnail
        self.size = self.wsi_wrapper.slide.dimensions

        # We have 6 actions, corresponding to "right", "up", "left", "down", zoom-in, zoom-out
        self.action_space = spaces.Discrete(6)
        # self.observation_space = Dict(
        #     {
        #         "current_view": Box(
        #             low=0, high=255, dtype="uint8"
        #         ),  # Variable-sized patch
        #         "birdeye_view": Box(
        #             low=0, high=255, dtype="uint8"
        #         ),  # Variable-sized thumbnail
        #     }
        # )

        self.observation_space = spaces.Tuple(
            [
                spaces.Box(low=0, high=255, dtype="uint8"),  # Variable-sized patch
                spaces.Box(low=0, high=255, dtype="uint8"),  # Variable-sized thumbnail
            ]
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([100, 0]),  # right
            1: np.array([0, -100]),  # up ( cuz of openslide it's flipped )
            2: np.array([-100, 0]),  # left
            3: np.array([0, 100]),  # down
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

        # self._agent_location = self.wsi_wrapper.position

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self.wsi_wrapper.render_current_info()

        return observation, info

    def step(self, discrete_action):
        # Map the action (element of {0,1,2,3,4,5}) to what we gonna do
        direction = self._action_to_direction[discrete_action]

        if isinstance(direction, np.ndarray):
            self.wsi_wrapper.move(*direction)
        else:
            if direction == -1:
                self.wsi_wrapper.zoom_in()
            else:
                self.wsi_wrapper.zoom_out()

        # An episode is done iff the agent has reached a view with a tissue_percentage < 0.1
        # print(self._get_info())
        current_tissue_percentage = helpers.get_tissue_percentage(
            self.wsi_wrapper.current_view
        )
        terminated = current_tissue_percentage < 0.1

        # reward = -1 if not terminated else 0  # Binary sparse rewards
        # We need to minimazie the tissue_percenentage
        reward = -current_tissue_percentage  # dense reward
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        # return {
        #     "current_view": np.asarray(self.wsi_wrapper.current_view, dtype=np.uint8),
        #     "birdeye_view": np.asarray(
        #         self.wsi_wrapper.current_bird_view, dtype=np.uint8
        #     ),
        # }
        return (
            np.asarray(self.wsi_wrapper.current_view, dtype=np.uint8),
            np.asarray(self.wsi_wrapper.current_bird_view, dtype=np.uint8),
        )

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
