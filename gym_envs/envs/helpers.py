import numpy as np
import cv2


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
