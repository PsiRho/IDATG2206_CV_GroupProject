import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(img, blurr_size):
    kernel = np.ones((blurr_size, blurr_size), np.float32)/blurr_size**2
    dst = cv2.filter2D(img, -1, kernel)

    return dst


def compare(org, new):
    max_dist = 282670692
    org_pixel_values = org.reshape(-1, 3).astype(np.int64)
    new_pixel_values = new.reshape(-1, 3).astype(np.int64)
    diff = 0
    for i in range(len(org_pixel_values)):
        diff += np.linalg.norm(org_pixel_values[i] - new_pixel_values[i])
    diff = 1 - np.round(diff / max_dist, 3)
    #print(1 - np.round(diff / max_dist, 3))
    return diff

def run_comp(org, new, blur_size = 3):
    org = gaussian_blur(org, blur_size)
    bad = gaussian_blur(new, blur_size)

    result = compare(org, bad)
    return result


