import matplotlib.pyplot as plt
import numpy as np
import cv2



#make image grayscale
#org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)


def histo(img: np.ndarray, bins: int = 256):
    """Compute histogram out of pixel values"""

    pix_values = np.zeros(bins, int)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pix_values[img[y, x]] += 1

    return pix_values

def compare_histo(org, new):
    diff = 0
    org_value = 0
    new_value = 0
    max_value = 0
    for i in range(len(org)):
        org_value += org[i]
        new_value += new[i]
        diff += abs(org[i]-new[i])
        max_value = new_value if new_value > org_value else org_value
    return diff/(max_value * 2)


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
    #plot_histo(org)
    print()



if __name__ == '__main__':
    main()
