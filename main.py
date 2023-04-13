import cv2
import threading
import numpy as np
import matplotlib.pyplot as plt
from MISS import histogram_diff as hd
from MISS import gaussian_comparisson as gc
from MISS import sobel_edge_detection as sed
from MISS import absolute_img_diff as absdiff
from MISS import comp_fft


# def compare_img():
#    """method for comparing the histogram for the two images"""
#
#    # dictionary for the compression types
#    compression_type = {
#        1: "1_JPEG2000_Compression",
#        2: "2_JPEG_Compression",
#        3: "3_Poisson_Noise",
#        4: "4_Gaussian_Blur",
#        5: "5_SGCK_Gamut_Mapping",
#        6: "6_DeltaE_Gamut_Mapping",
#    }
#
#    correlation = []
#
#    # loop through the compression types
#    for compression in range(1, 7):
#        # loop through the pictures
#        for picture in range(1, 24):
#            # making sure the picture number is 2 digits
#            if picture < 10:
#                picture = f"0{picture}"
#            # making path to the original image
#            org = cv2.imread(f'CIDIQ_Dataset/Images/Original/final{picture}.bmp')
#            for i in range(5):
#                # making path to the image to compare with original
#                path = f'CIDIQ_Dataset/Images/reproduction/{compression_type[compression]}/final{picture}_d{compression}_l{i + 1}.bmp'
#                # reading the image
#                new = cv2.imread(path)
#                # getting the histogram correlation
#                diff = get_diff(org, new)
#                # appending the correlation to the list
#                correlation.append(diff)
#                # printing the correlation
#                print(f'{compression_type[compression]} : picture {picture} : level {i + 1} : {diff}')
#                # print(f'final{picture}_d{compression}_l{i + 1}.bmp : {diff}')
#    print(correlation)


def get_diff2(org, new):
    """method for getting the difference between the two images"""
    print("-------------------------------------------------------")

    diff1 = gc.run_comp(org, new)
    print(f"mean blur difference = {diff1}")

    diff2 = hd.compare_binging_hist_correlation(org, new)
    print(f"histogram difference = {diff2}")

    diff3 = sed.get_score(org, new)
    print(f"sobel difference = {diff3}")

    diff4 = absdiff.absolute_img_diff(org, new)
    print(f"absolute difference = {diff4}")

    return round((diff1 + diff2 + diff3 + diff4) / 4, 4)


def get_diff(org, new):
    """method for getting the difference between the two images"""

    results = {}

    # Define functions to run on separate threads
    def run_mean_blur():
        diff1 = gc.run_comp(org, new)
        print(f"mean blur difference = {diff1}")
        results['diff1'] = diff1

    def run_histogram():
        diff2 = hd.compare_binging_hist_correlation(org, new)
        print(f"histogram difference = {diff2}")
        results['diff2'] = diff2

    def run_sobel_diff():
        diff3 = sed.get_score(org, new)
        print(f"sobel difference = {diff3}")
        results['diff3'] = diff3

    def run_absolute_diff():
        diff4 = absdiff.absolute_img_diff(org, new)
        print(f"absolute difference = {diff4}")
        results['diff4'] = diff4

    def run_fft():
        diff5 = comp_fft.compare_fft(org, new)
        print(f"fft difference = {diff5}")
        results['diff5'] = diff5

    # Run the functions on separate threads
    threads = [
        threading.Thread(target=run_mean_blur),
        threading.Thread(target=run_histogram),
        threading.Thread(target=run_sobel_diff),
        threading.Thread(target=run_absolute_diff),
        threading.Thread(target=run_fft)
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Calculate the average difference
    meanblurdiff = results['diff1']
    histdiff = results['diff2']
    sobeldiff = results['diff3']
    absolutediff = results['diff4']
    fftdiff = results['diff5']

    meanblurdiff_weight = 0.2
    histdiff_weight = 0.5
    sobeldiff_weight = 0.5
    absdiff_weight = 0.5
    fftdiff_weight = 1

    weighted_sum = (meanblurdiff * meanblurdiff_weight) + \
                   (histdiff * histdiff_weight) + \
                   (sobeldiff * sobeldiff_weight) + \
                   (absolutediff * absdiff_weight) + \
                   (fftdiff * fftdiff_weight)

    avg_diff = weighted_sum / 5

    return round(avg_diff, 3)
