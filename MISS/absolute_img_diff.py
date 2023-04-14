import numpy as np


def absolute_img_diff(img1, img2):

    # Make sure the images are the same shape
    assert img1.shape == img2.shape, "Images must be the same shape."

    # Calculate the absolute difference between the two images
    diff = np.ndarray.sum(np.abs(img1 - img2))

    # Normalize to get a value between 0 and 1
    norm_diff = diff / (img1.shape[0] * img1.shape[1] * img1.shape[2] * 255)

    return round(norm_diff, 4)
