import numpy as np
import cv2
import matplotlib.pyplot as plt

orig_img = cv2.imread('../CIDIQ_Dataset/Images/Original/final01.bmp')
degraded_img = cv2.imread('../CIDIQ_Dataset/Images/Reproduction/1_JPEG2000_Compression/final01_d1_l1.bmp')

orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)


def absolute_img_diff(img1, img2):
    return np.ndarray.sum(np.abs(img1 - img2))



def histogram_comparison(img1, img2):
    """method for comparing the histogram for the two images"""
    org_hist, edge1 = np.histogram(img1, bins=np.arange(255))
    shit_hist, edge2 = np.histogram(img2, bins=np.arange(255))



    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(org_hist)
    plt.subplot(2, 1, 2)
    plt.plot(shit_hist)
    plt.show()
    cv2.compareHist(org_hist, shit_hist, cv2.HISTCMP_CORREL)

    return edge1, edge2


def edge_detection(img1, img2):
    return np.ndarray.sum(np.abs(cv2.Canny(img1) - cv2.Canny(img2)))


def kernel_comparison(img1, img2):
    return np.ndarray.sum(np.abs(cv2.filter2D(img1) - cv2.filter2D(img2)))

print("test1")
print(histogram_comparison(orig_img, degraded_img))
print("test2")

