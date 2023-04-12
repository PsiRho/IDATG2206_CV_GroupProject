import numpy as np

'''
A function for finding the ideal threshold of an image using otsu's algorithm
    Parameters:
        - img - the image of which to find the threshold
        - bins  - the amount of bins for the histogram (more = more accurate)  
'''


def otsus(img, bins):
    # Get the image histogram and the bin edges(Would implement own histogram function, but it is hard to get bin edges in that case).
    histogram, edges = np.histogram(img, bins=bins)

    # Calculate bins centers (values)
    centres = (edges[:-1] + edges[1:]) / 2.

    # Iterate over all thresholds and get the weights
    weight1 = np.cumsum(histogram)
    weight2 = np.cumsum(histogram[::-1])[::-1]

    # Get the class means
    mean1 = np.cumsum(histogram * centres) / weight1
    mean2 = (np.cumsum((histogram * centres)[::-1]) / weight2[::-1])[::-1]

    # Get class variance
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function value
    index_of_max_val = np.argmax(inter_class_variance)

    # Calculate threshold for image
    thresh = centres[:-1][index_of_max_val]

    return thresh
