import numpy as np


def compare_hist_correlation(img1, img2):
    """
    Compute correlation between two histograms. If the image is grayscale, the np.histogram() function will be used.
    If the image is color, the np.histogramdd() function will be used. The histograms will have 256 bins.

    The histograms are normalized so that they sum to 1. This is done by dividing each bin value by the total number of
    pixels in the image. This is done to make sure the correlation calculation is based on shape and distribution of the
    histograms, not the total number of pixels in the image.

    The correlation is computed as the sum of the product of the normalized histograms
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

    assert img1.shape == img2.shape, "Images must be the same shape."

    # Compute histograms of the two images
    if len(img1.shape) == 3:
        hist1, _ = np.histogramdd(img1.reshape(-1, 3), bins=256, range=[(0, 255), (0, 255), (0, 255)])
        hist2, _ = np.histogramdd(img2.reshape(-1, 3), bins=256, range=[(0, 255), (0, 255), (0, 255)])
    else:
        hist1, _ = np.histogram(img1, bins=256, range=[0, 256])
        hist2, _ = np.histogram(img2, bins=256, range=[0, 256])

    # Normalize histograms
    hist1_norm = hist1 / np.sum(hist1)
    hist2_norm = hist2 / np.sum(hist2)

    # Compute correlation between the histograms
    correlation = np.sum((hist1_norm - np.mean(hist1_norm)) * (hist2_norm - np.mean(hist2_norm)))
    correlation /= (np.std(hist1_norm) * np.std(hist2_norm))

    return correlation
