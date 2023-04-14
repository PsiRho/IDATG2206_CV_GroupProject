import cv2
import threading
import numpy as np
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from MISS import histogram_diff as histdiff
from MISS import gaussian_comparisson as meanblur
from MISS import sobel_edge_detection as sobel
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

    diff1 = meanblur.run_comp(org, new)
    print(f"mean blur difference = {diff1}")

    diff2 = histdiff.compare_binging_hist_correlation(org, new)
    print(f"histogram difference = {diff2}")

    diff3 = sobel.get_score(org, new)
    print(f"sobel difference = {diff3}")

    diff4 = absdiff.absolute_img_diff(org, new)
    print(f"absolute difference = {diff4}")

    return round((diff1 + diff2 + diff3 + diff4) / 4, 4)


def get_diff2(org, new):
    """method for getting the difference between the two images"""

    results = {}

    # Define functions to run on separate threads
    def run_mean_blur():
        diff1 = meanblur.run_comp(org, new)
        print(f"mean blur difference = {diff1}")
        results['diff1'] = diff1

    def run_histogram():
        diff2 = histdiff.compare_binging_hist_correlation(org, new)
        print(f"histogram difference = {diff2}")
        results['diff2'] = diff2

    def run_sobel_diff():
        diff3 = sobel.get_score(org, new)
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
    absdiff_weight = 0.7
    fftdiff_weight = 0.2

    weighted_sum = (meanblurdiff * meanblurdiff_weight) + \
                   (histdiff * histdiff_weight) + \
                   (sobeldiff * sobeldiff_weight) + \
                   (absolutediff * absdiff_weight) + \
                   (fftdiff * fftdiff_weight)

    avg_diff = weighted_sum / 5

    return round(avg_diff, 3)


def run_mean_blur(org, new, queue):
    diff1 = meanblur.run_comp(org, new)
    print(f"mean blur difference = {diff1}")
    queue.put(('diff1', diff1))


def run_histogram(org, new, queue):
    diff2 = histdiff.compare_binging_hist_correlation(org, new)
    print(f"histogram difference = {diff2}")
    queue.put(('diff2', diff2))


def run_sobel_diff(org, new, queue):
    diff3 = sobel.get_score(org, new)
    print(f"sobel difference = {diff3}")
    queue.put(('diff3', diff3))


def run_absolute_diff(org, new, queue):
    diff4 = absdiff.absolute_img_diff(org, new)
    print(f"absolute difference = {diff4}")
    queue.put(('diff4', diff4))


def run_fft(org, new, queue):
    diff5 = comp_fft.compare_fft(org, new)
    print(f"fft difference = {diff5}")
    queue.put(('diff5', diff5))


def calculate_weighted_average(results, weights):
    weighted_sum = sum(value * weight for value, weight in zip(results, weights))
    avg_diff = weighted_sum / sum(weights)
    return round(avg_diff, 3)


def get_diff(org, reprod):
    queue = Queue()
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.submit(run_mean_blur, org, reprod, queue)
        executor.submit(run_histogram, org, reprod, queue)
        executor.submit(run_sobel_diff, org, reprod, queue)
        executor.submit(run_absolute_diff, org, reprod, queue)
        executor.submit(run_fft, org, reprod, queue)

    results = {}
    while not queue.empty():
        key, value = queue.get()
        results[key] = value

    weights = [0.2, 0.5, 0.5, 0.7, 0.2]
    results_list = [results['diff1'], results['diff2'], results['diff3'], results['diff4'], results['diff5']]
    weighted_average = calculate_weighted_average(results_list, weights)
    print(f"Weighted Average Difference: {weighted_average}")
    return weighted_average
