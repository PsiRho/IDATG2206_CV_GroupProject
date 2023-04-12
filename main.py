import cv2
import numpy as np
import matplotlib.pyplot as plt
from MISS import histogram_diff as hd
from MISS import gaussian_comparisson as gc

def compare_img():
    """method for comparing the histogram for the two images"""

    compression_type = {
        1: "1_JPEG2000_Compression",
        2: "2_JPEG_Compression",
        3: "3_Poisson_Noise",
        4: "4_Gaussian_Blur",
        5: "5_SGCK_Gamut_Mapping",
        6: "6_DeltaE_Gamut_Mapping",
    }

    correlation = []

    # loop through the compression types
    for compression in range(1, 7):
        # loop through the pictures
        for picture in range(1, 24):
            # making sure the picture number is 2 digits
            if picture < 10:
                picture = f"0{picture}"
            # making path to the original image
            org = cv2.imread(f'CIDIQ_Dataset/Images/Original/final{picture}.bmp')
            for i in range(5):
                # making path to the image to compare with original
                path = f'CIDIQ_Dataset/Images/reproduction/{compression_type[compression]}/final{picture}_d{compression}_l{i + 1}.bmp'
                # reading the image
                new = cv2.imread(path)
                # getting the histogram correlation
                diff1, diff2 = get_diff(org, new)
                # appending the correlation to the list
                correlation.append(diff2)
                # printing the correlation
                print(f'final{picture}_d{compression}_l{i + 1}.bmp : {diff2}')
    print(correlation)


def get_diff(org, new):
    diff1 = hd.compare_hist_correlation(org, new)
    diff2 = gc.compare_gaussian(org, new)
    return diff1, diff2

def main():
  compare_img()

if __name__ == '__main__':
    main()