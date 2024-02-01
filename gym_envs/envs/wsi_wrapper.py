import os

if os.name == "nt":
    _dll_path = os.getenv("OPENSLIDE_PATH")
    if _dll_path is not None:
        with os.add_dll_directory(_dll_path):
            import openslide
    else:
        import openslide
else:
    import openslide


import numpy as np

from . import helpers
from PIL import ImageDraw


class WSIApi:
    """
    An Api for a Whole Slide Image (WSI) that implements the controls.

    Parameters:
    - wsi_path (str): The path to the WSI file.
    - patch_size (tuple[int]): The size of the patches to extract from the WSI. Defaults to (128,128).
    - resize_thumbnail (tuple[int] or bool): The size to resize the thumbnail to. If set to True, the thumbnail will be resized to (512, 512). Defaults to (512, 512).
    - threshold (int): The threshold value used for tissue detection. Defaults to 200.

    Methods:
    - render_current_info(with_bird_view=False): Renders the current position and view. If with_bird_view is True, it also shows the bird's eye view.
    - get_view(position, level, patch_size): Gets the view at the specified position, level, and patch size.
    - move(dx, dy): Moves the agent by dx and dy while staying in the same level.
    - zoom_in(dx=0, dy=0): Zooms in gradually to the next magnification level. The optional dx and dy specify the position to zoom in.
    - zoom_out(dx=0, dy=0): Zooms out gradually to the previous magnification level. The optional dx and dy specify the position to zoom out.
    - get_current_bird_view(outline_color="red", rect_width=2): Gets the bird's eye view of the current view, with the agent's position denoted by a red rectangle.
    - reset(seed=None, options=None): Resets the agent to a random position within the tissue area of the WSI.

    Attributes:
    - slide: The openslide object representing the WSI.
    - thumbnail: The thumbnail image of the WSI.
    - max_allowed_level: The maximum level of zoom-out.
    - current_view: The current view of the WSI.
    - position: The current position of the agent.
    - level: The current zoom level.
    - patch_size: The size of the patches.
    - resize_thumbnail: The size to resize the thumbnail to.
    - threshold: The threshold value used for tissue detection.
    - maxx: The maximum x-coordinate of the tissue area.
    - maxy: The maximum y-coordinate of the tissue area.
    - mapped_x: The mapped x-coordinate of the tissue area.
    - mapped_y: The mapped y-coordinate of the tissue area.
    - bird_position: The position of the agent in the bird's eye view.
    - bird_view_size: The size of the agent's view in the bird's eye view.
    - current_bird_view: The current bird's eye view of the WSI.
    - thumbnail_size: The size of the thumbnail image.
    """

    def __init__(
        self,
        wsi_path,
        patch_size=(128, 128),
        resize_thumbnail=(512, 512),
        threshold=200,
    ):
        """
        Initializes the WSIWrapper object.

        Args:
            wsi_path (str): The path to the Whole Slide Image (WSI) file.
            patch_size (tuple, optional): The size of the patches to extract from the WSI. Defaults to (128, 128).
            resize_thumbnail (tuple or bool, optional): The size to resize the thumbnail image to. If set to True, the thumbnail will not be resized. Defaults to (512, 512).
            threshold (int, optional): The threshold value for tissue detection. Defaults to 200.
        """
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        if not isinstance(resize_thumbnail, bool) and isinstance(
            resize_thumbnail, (np.integer, int)
        ):
            resize_thumbnail = (resize_thumbnail, resize_thumbnail)

        self.slide = openslide.open_slide(wsi_path)

        # construct thumbnail and setting the max zoom-out level
        self.__get_best_level_for_thumbnail()  # This defines self.thumbnail and self.max_allowed_level
        self.current_view = None

        self.level = 0
        self.patch_size = patch_size
        self.resize_thumbnail = resize_thumbnail
        self.threshold = threshold

        # set a random position in the WSI in tissue.
        # helper variables to not calculate the same thing over and over again
        x, y, w, h = helpers.detect_tissue_thumbnail(self.thumbnail, threshold)
        downsample_factor = int(self.slide.level_downsamples[self.max_allowed_level])

        self.mapped_x, self.mapped_y = x * downsample_factor, y * downsample_factor
        mapped_w, mapped_h = w * downsample_factor, h * downsample_factor

        self.maxx, self.maxy = self.mapped_x + mapped_w, self.mapped_y + mapped_h

        self.__get_random_position_in_tissue()

    def render_current_info(self, with_bird_view=False):
        print(self.position)
        self.current_view.show()

        if with_bird_view:
            self.current_bird_view.show()

    def get_view(self, position, level, patch_size):
        # TODO: Think about if we should modify the logic of position depending on the level ? (x,y) is always linked to level 0 and top-left

        # replaced by np.clip
        # assert 0 <= x < width and 0 <= y < height
        # assert 0 <= level < self.max_allowed_level # defined in __get_best_level_for_thumbnail()

        self.current_view = self.slide.read_region(position, level, patch_size).convert(
            "RGB"
        )
        self.position = position
        self.level = level
        self.patch_size = patch_size

        # update the bird view
        self.get_current_bird_view()

    def move(self, dx, dy):
        """
        Move the agent dx, dy while staying in the same level
        """
        self.__update_position(dx, dy)

        self.get_view(self.position, self.level, self.patch_size)

    def zoom_in(self, dx=0, dy=0):
        """
        Not sure if we should do it gradually or allow jumping between different levels ?
        We will do it gradually first.

        (dx, dy) will specify where we exactly wanna zoom in. A zoom with default will just zoom on the top left part of the current view.
        """
        if self.level == 0:
            # print("Already in the max magnification, can't zoom-in more. Returning")
            return

        self.level = np.clip(self.level, 0, self.max_allowed_level - 1)

        self.level -= 1  # gradually to the next magnification
        self.__update_position(dx, dy)  # getting where we wanna exactly zoom in
        self.get_view(self.position, self.level, self.patch_size)

    def zoom_out(self, dx=0, dy=0):
        """
        Not sure if we should do it gradually or allow jumping between different levels ?
        We will do it gradually first.

        (dx, dy) will specify where we exactly wanna zoom in. A zoom with default will just zoom out from the top left part of the current view.
        """
        if self.level == self.max_allowed_level:
            # print("Already in the min magnification, can't zoom-out more. Returning")
            return

        self.level = np.clip(self.level, 0, self.max_allowed_level - 1)

        self.level += 1  # gradually to the next magnification
        self.__update_position(dx, dy)  # getting where we wanna exactly zoom out
        self.get_view(self.position, self.level, self.patch_size)

    def get_current_bird_view(self, outline_color="red", rect_width=2):
        """
        Locating where the agent is in the thumbnail of the current slide.
        Will always take the self.current_view and return current_view denoted by a red rectangle on the thumbnail of the WSI.

        """

        # mapping the coords
        (
            x0,
            y0,
        ) = (
            self.position
        )  # coords are always in the level 0, hence downsample factor will always be mapped to the max level
        downsample_factor = int(self.slide.level_downsamples[self.max_allowed_level])

        x_thumb_level, y_thumb_level = x0 * (1 / downsample_factor), y0 * (
            1 / downsample_factor
        )
        x_thumb_level, y_thumb_level = int(x_thumb_level), int(y_thumb_level)

        # mapping the size of the patch to the thumbnail
        w, h = self.patch_size

        # w,h needs to be adjusted according to the current zoom level hence the minus self.level
        downsample_factor = int(
            self.slide.level_downsamples[self.max_allowed_level - self.level]
        )
        w_thumb_level, h_thumb_level = w * (1 / downsample_factor), h * (
            1 / downsample_factor
        )

        # construct the thumbnail and drawing the rectangle of the current field of view of the agent
        thumbnail = self.thumbnail.copy()
        draw = ImageDraw.Draw(thumbnail)
        draw.rectangle(
            [
                x_thumb_level,
                y_thumb_level,
                x_thumb_level + w_thumb_level,
                y_thumb_level + h_thumb_level,
            ],
            outline=outline_color,
            width=rect_width,
        )

        # maybe only these two will be given to the agent, along side with the original thumbnail (self.thumbnail) no need to copy it
        self.bird_position = (x_thumb_level, y_thumb_level)
        self.bird_view_size = (w_thumb_level, h_thumb_level)

        if self.resize_thumbnail:
            # calculate the scaling to map the rect info to the new size birdview
            W, H = np.asarray(thumbnail).shape[:2]
            S_W, S_H = self.resize_thumbnail
            scale_x, scale_y = S_W / W, S_H / H

            # update the coords
            self.bird_position = (x_thumb_level * scale_x, y_thumb_level * scale_y)
            self.bird_view_size = (w_thumb_level * scale_x, h_thumb_level * scale_y)

            thumbnail = thumbnail.resize(self.resize_thumbnail)

        # set the bird view
        self.current_bird_view = thumbnail

        thumbnail_shape = np.asarray(thumbnail).shape
        self.thumbnail_size = (thumbnail_shape[0], thumbnail_shape[1])

    def __get_best_level_for_thumbnail(self, min=1000, max=2000):
        """
        Will get the level in which the avg of its dimensions ((w+h)/2) is between min and max.
        That will be our thumbnail where it's gonna be used to have a bird view.
        Also that will be the max level of zoom-out
        """
        avgs = np.asarray([int(np.mean(dim)) for dim in self.slide.level_dimensions])
        level = np.where((avgs >= min) & (avgs <= max))[0].item()

        self.thumbnail = self.slide.read_region(
            (0, 0), level, self.slide.level_dimensions[level]
        ).convert("RGB")
        self.max_allowed_level = level

    def __update_position(self, dx, dy):
        old_x, old_y = self.position

        # We use `np.clip` to make sure we don't leave the slide
        new_x = np.clip(old_x + dx, 0, self.slide.dimensions[0] - 1)
        new_y = np.clip(old_y + dy, 0, self.slide.dimensions[1] - 1)

        self.position = (new_x, new_y)

    def __get_random_position_in_tissue(self, seed=None):
        """
        Returns a random position within the tissue region.

        Args:
            seed (int, optional): Seed value for reproducibility. Defaults to None.

        Returns:
            tuple: Random position within the tissue region.
        """
        if seed is not None:
            np.random.seed(seed)

        random_x = np.random.randint(self.mapped_x, self.maxx)
        random_y = np.random.randint(self.mapped_y, self.maxy)

        position = (random_x, random_y)
        self.get_view(position, self.level, self.patch_size)
        percentage_tissue = helpers.get_tissue_percentage(self.current_view)

        while percentage_tissue < 0.5:
            random_x = np.random.randint(self.mapped_x, self.maxx)
            random_y = np.random.randint(self.mapped_y, self.maxy)

            position = (random_x, random_y)
            self.get_view(position, self.level, self.patch_size)

            percentage_tissue = helpers.get_tissue_percentage(self.current_view)

        print("Started in position: ", position)
        return position

    def reset(self, seed=None, options=None):
        return self.__get_random_position_in_tissue(seed=seed)
