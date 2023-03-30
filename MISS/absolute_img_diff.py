import numpy as np


def absolute_img_diff(img1, img2):
    return np.ndarray.sum(np.abs(img1 - img2))
