import matplotlib.pyplot as plt
import numpy as np
import cv2


# make image grayscale
# org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)


def histo(img: np.ndarray, bins: int = 256):
    """Compute histogram out of pixel values"""

    pix_values = np.zeros(bins, int)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pix_values[img[y, x]] += 1

    return pix_values


def compare_histo(org, new):
    """method for comparing the difference between two histograms"""

    diff = 0
    org_value = 0
    new_value = 0
    max_value = 0
    # loop through the histograms
    for i in range(len(org)):
        # Add up the total for both histograms
        org_value += org[i]
        new_value += new[i]
        # get the absolute value for the difference between the histograms
        diff += abs(org[i] - new[i])
        # set the highest values to be the max values
        max_value = new_value if new_value > org_value else org_value

    # return the difference between the two histograms divided by the twice the max value
    # to get a value between 0 and 1, which is the percentage of difference
    return 1 - (diff / (max_value * 2))


def compare_hist_correlation(img1, img2):
    """
    Compute correlation between two histograms. The np.histogram() function is used to compute the histograms.
    It has 256 bins, which correspond to the 256 possible pixel intensity values. The histograms are normalized
    so that they sum to 1. The correlation is computed as the sum of the product of the normalized histograms
    minus the mean of the normalized histograms, divided by the product of the standard deviation of the normalized
    histograms.

    The general formula for correlation is: sum((x - mean(x)) * (y - mean(y))) / (std(x) * std(y))
    where x and y are the two histograms.

    If the histograms are identical, the correlation will be 1. If the histograms are completely
    different, the correlation will be 0.

    :param img1: an image to compare
    :param img2: an image to compare
    :return: the correlation between the two histograms as a float between 0 and 1 where 1 is identical and 0 is
    completely different.
    """
    # Compute histograms of the two images
    hist1, _ = np.histogram(img1, bins=256, range=[0, 256])
    hist2, _ = np.histogram(img2, bins=256, range=[0, 256])

    # Normalize histograms
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)

    # Compute correlation between the histograms
    correlation = np.sum((hist1_norm - np.mean(hist1_norm)) * (hist2_norm - np.mean(hist2_norm)))
    correlation /= (np.std(hist1_norm) * np.std(hist2_norm))

    return correlation


def plot_histo(a, img: np.ndarray):
    """Plot histogram out of vector"""
    plt.plot(a)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")
    plt.show()


def main():
    # Read image
    org = cv2.imread('../CIDIQ_Dataset/Images/Original/final01.bmp')
    new = cv2.imread('../CIDIQ_Dataset/Images/Original/final02.bmp')

    b = histo(new)
    a = histo(org)
    diff = compare_histo(a, b)
    print(diff)
    # plot_histo(org)
    print()


if __name__ == '__main__':
    main()
