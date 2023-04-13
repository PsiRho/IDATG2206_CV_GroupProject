import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(img, blurr_size):
    """Apply gaussian blur to image"""
    kernel = np.ones((blurr_size, blurr_size), np.float32) / blurr_size ** 2
    dst = cv2.filter2D(img, -1, kernel)

    return dst


def compare(org, new):
    """Compare two images using pixel values"""
    org_pixel_values = org.reshape(-1, 3).astype(np.int64)
    new_pixel_values = new.reshape(-1, 3).astype(np.int64)

    max_value = np.amax(np.abs(org_pixel_values - new_pixel_values))

    diff = 0

    abs_diff = np.abs(org_pixel_values - new_pixel_values)
    abs_diff = 1 - np.mean(abs_diff / 255)
    """ print(abs_diff)
    for i in range(len(org_pixel_values)):
        diff += np.linalg.norm(org_pixel_values[i] - new_pixel_values[i])
    diff = 1 - np.round(diff / max_dist, 3)
    #print(1 - np.round(diff / max_dist, 3))
    print(diff)"""
    return abs_diff


def run_comp(org, new, blur_size=3):
    org = gaussian_blur(org, blur_size)
    bad = gaussian_blur(new, blur_size)

    result = compare(org, bad)
    return round(result, 3)



compare(cv2.imread("C:\\Users\\johan\\Desktop\\MISS\\images\\test\\test1.jpg"), cv2.imread("C:\\Users\\johan\\Desktop\\MISS\\images\\test\\test2.jpg"))