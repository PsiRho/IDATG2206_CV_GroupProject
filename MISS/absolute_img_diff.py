import numpy as np


def absolute_img_diff(img1, img2):
    # Calculate the absolute difference between the two images
    diff = np.ndarray.sum(np.abs(img1 - img2))

    # Normalize to get a value between 0 and 1
    norm_diff = diff / (img1.shape[0] * img1.shape[1] * img1.shape[2] * 255)

    return round(norm_diff, 4)


# def histogram_comparison(img1, img2):
#    """method for comparing the histogram for the two images"""
#    org_hist, edge1 = np.histogram(img1, bins=np.arange(255))
#    shit_hist, edge2 = np.histogram(img2, bins=np.arange(255))
#
#
#
#    plt.figure()
#    plt.subplot(2, 1, 1)
#    plt.plot(org_hist)
#    plt.subplot(2, 1, 2)
#    plt.plot(shit_hist)
#    plt.show()
#    cv2.compareHist(org_hist, shit_hist, cv2.HISTCMP_CORREL)
#
#    return edge1, edge2
#
#
# def edge_detection(img1, img2):
#    return np.ndarray.sum(np.abs(cv2.Canny(img1) - cv2.Canny(img2)))
#
#
# def kernel_comparison(img1, img2):
#    return np.ndarray.sum(np.abs(cv2.filter2D(img1) - cv2.filter2D(img2)))
#
# print("test1")
# print(histogram_comparison(org, shit))
# print("test2")
