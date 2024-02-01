import openslide
import numpy as np
import cv2
import h5py
import itertools
import heapq

from typing import Tuple
from tqdm import trange


def get_random_position(slide_dimensions):
    width, height = slide_dimensions
    x = np.random.randint(width)
    y = np.random.randint(height)

    return (x, y)


def detect_tissue_thumbnail(thumbnail, threshold=200):
    # Convert the thumbnail to a NumPy array
    thumbnail_np = np.asarray(thumbnail)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)

    # Threshold the image to identify potential tissue regions
    _, binary_image = cv2.threshold(
        grayscale_image, threshold, 255, cv2.THRESH_BINARY_INV
    )

    # Perform morphology operations (closing) to enhance the tissue regions
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(
        closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)


def generate_tissue_mask(thumbnail, threshold=200):
    # Convert the thumbnail to a NumPy array
    thumbnail_np = np.array(thumbnail)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)

    # Threshold the image to identify potential tissue regions
    _, binary_image = cv2.threshold(
        grayscale_image, threshold, 255, cv2.THRESH_BINARY_INV
    )

    # Perform morphology operations (closing) to enhance the tissue regions
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return closed_image / 255


def generate_random_coords_in_mask(mask, patch_size, tissue_percentage_threshold=80):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty mask for the tissue region
    tissue_mask = np.zeros_like(mask)

    # Iterate through contours and fill the tissue mask
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (tissue_percentage_threshold / 100) * patch_size[0] * patch_size[1]:
            cv2.drawContours(tissue_mask, [contour], 0, 255, thickness=cv2.FILLED)

    # Find coordinates where tissue is present in the mask
    tissue_coords = np.column_stack(np.where(tissue_mask > 0))

    # Randomly select coordinates from the tissue region
    random_coords = tissue_coords[
        np.random.choice(tissue_coords.shape[0], size=1, replace=False)
    ]

    return tuple(random_coords[0])


def get_tissue_percentage(img):
    mask = generate_tissue_mask(img)
    w, h = mask.shape
    surface = w * h
    percentage_tissue = len(np.where(mask == 1)[0]) / surface
    return percentage_tissue


def binary_encoding(number, n_bits=4) -> list[int]:
    # given a number, and n_bits sized array, return the binary representation of number
    if number < 0 or number > 2**n_bits:
        raise ValueError(
            f"Can't encode {number} in a binary format using just {n_bits} bits"
        )

    binary_representation = format(number, f"0{n_bits}b")
    return [int(bit) for bit in binary_representation]


def map_coords_to_center_and_normalize(slide_dimensions, x, y) -> Tuple[float, float]:
    # map the coords of the patch (x, y) of openslide to the central point of the WSI
    # coords here will be in the range of [-1, 1]
    width, height = slide_dimensions  # Level 0 dimensions
    center_x, center_y = width / 2, height / 2

    normalized_x = (x - center_x) / width
    normalized_y = (y - center_y) / height

    return normalized_x, normalized_y


def map_rect_info(
    thumbnail_shape: Tuple[int, int], x: int, y: int, width: int, height: int
) -> Tuple[float, float, float, float]:
    """
    Given the rect that is drawn on the thumbnail that shows where the current view is in the WSI.
    Get the mapped info: (normalized_x, normalized_y, percentage_width, percentage_height)
    """

    normalized_x, normalized_y = map_coords_to_center_and_normalize(
        thumbnail_shape, x, y
    )
    percentage_width, percentage_height = (
        width / thumbnail_shape[0],
        height / thumbnail_shape[1],
    )

    return (normalized_x, normalized_y, percentage_width, percentage_height)


def create_attention_score_map_l0(
    slide_path, hdf5_path, patch_size, normalize=True
) -> np.ndarray[np.float32]:
    with openslide.OpenSlide(slide_path) as slide:
        dx, dy = patch_size
        slide_dim = slide.dimensions

        with h5py.File(hdf5_path, "r") as h5:
            minimum_value = np.asarray(h5["attention_scores"]).min()
            FILL_VALUE = minimum_value * 10  # probably for the background

            attention_scores = np.full(
                slide_dim[::-1], FILL_VALUE, dtype=np.float32
            )  # Reverse dimensions
            print(
                f"Creating an attention scores array of shape {slide_dim[::-1]} filled with {FILL_VALUE}"
            )

            coords = h5["coords"][:]
            scores = h5["attention_scores"][:]
            for i in range(len(coords)):
                x, y = coords[i]
                score = scores[i]
                try:
                    if np.all(attention_scores[y : y + dy, x : x + dx] == FILL_VALUE):
                        attention_scores[y : y + dy, x : x + dx] = score
                    else:
                        # Average the existing and new value
                        attention_scores[y : y + dy, x : x + dx] = (
                            attention_scores[y : y + dy, x : x + dx] + score
                        ) / 2

                except IndexError:
                    print(f"IndexError at coordinates: {coords[i]}")

            if normalize:
                min_score = FILL_VALUE
                max_score = scores.max()

                attention_scores = (attention_scores - min_score) / (
                    max_score - min_score
                )
                attention_scores = 2 * attention_scores - 1

    # Optimize code to make it faster
    attention_scores = attention_scores.astype(np.float32)

    return attention_scores


def normalize_scores(scores) -> np.ndarray[np.float32]:
    # Ensure scores is a numpy array
    scores = np.asarray(scores)

    # Find the minimum and maximum values in the scores
    min_score = np.min(scores)
    max_score = np.max(scores)

    # Normalize the scores to a 0-1 range
    normalized_scores = (scores - min_score) / (max_score - min_score)

    # Scale the normalized scores to a -1 to 1 range
    normalized_scores = 2 * normalized_scores - 1

    return normalized_scores


class PatchPriorityQueue:
    def __init__(self, capacity=10):
        self.heap = []
        self.capacity = capacity
        self.counter = itertools.count()  # Tie-breaker counter

    def add_patch_metadata(self, attention_score, patch_metadata):
        # Use counter as a tie-breaker for equal scores
        count = next(self.counter)
        heapq.heappush(self.heap, (-attention_score, count, patch_metadata))

        if len(self.heap) > self.capacity:
            heapq.heappop(self.heap)

    def get_patches(self):
        return [data for score, count, data in sorted(self.heap, reverse=True)]
