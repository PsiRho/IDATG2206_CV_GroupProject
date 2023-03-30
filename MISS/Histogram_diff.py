import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read image
org = cv2.imread('../CIDIQ_Dataset/Images/Original/final01.bmp')

#make image grayscale
#org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)


def histo(img: np.ndarray, bins: int = 256):
    """Compute histogram out of pixel values"""

    pix_values = np.zeros(bins, int)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pix_values[img[y, x]] += 1

    return pix_values


def plot_histo(img: np.ndarray):
    """Plot histogram out of vector"""
    a = histo(org)
    plt.plot(a)
    plt.xlabel("pixel value")
    plt.ylabel("number of pixels")
    plt.show()



def main():
    plot_histo(org)



if __name__ == '__main__':
    main()
