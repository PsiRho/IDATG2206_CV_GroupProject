import cv2
import numpy as np
import matplotlib.pyplot as plt
from MISS import histogram_diff as hd
from MISS import gaussian_comparisson as gc
from MISS import sobel_edge_detection as sed

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
    for compression in range(6, 7):
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
                diff = get_diff(org, new)
                # appending the correlation to the list
                correlation.append(diff)
                # printing the correlation
                print(f'final{picture}_d{compression}_l{i + 1}.bmp : {diff}')
    print(correlation)


def get_diff(org, new):
    print()
    diff1 = gc.run_comp(org, new)
    print(f"gaussian difference = {diff1}")
    diff2 = hd.compare_hist_correlation(org, new)
    print(f"histogram difference = {diff2}")
    diff3 = sed.get_score(org, new)
    print(f"sobel difference = {diff3}")
    return round((diff1 + diff2 + diff3) / 3, 3)

def main():
  compare_img()

if __name__ == '__main__':
    main()