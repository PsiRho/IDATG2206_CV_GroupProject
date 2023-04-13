import cv2
import threading
import numpy as np
import matplotlib.pyplot as plt
from MISS import histogram_diff as hd
from MISS import gaussian_comparisson as gc
from MISS import sobel_edge_detection as sed


def compare_img():
    """method for comparing the histogram for the two images"""

    # dictionary for the compression types
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
                diff = get_diff(org, new)
                # appending the correlation to the list
                correlation.append(diff)
                # printing the correlation
                print(f'{compression_type[compression]} : picture {picture} : level {i + 1} : {diff}')
                # print(f'final{picture}_d{compression}_l{i + 1}.bmp : {diff}')
    print(correlation)


def get_diff(org, new):
    """method for getting the difference between the two images"""
    print()
    diff1 = gc.run_comp(org, new)
    print(f"gaussian difference = {diff1}")
    diff2 = hd.compare_binging_hist_correlation(org, new)
    print(f"histogram difference = {diff2}")
    diff3 = sed.get_score(org, new)
    print(f"sobel difference = {diff3}")
    return round((diff1 + diff2 + diff3) / 3, 3)



def get_threaded_diff(org, new):
    """method for getting the difference between the two images"""
    print()

    # Definer funksjonene som skal kjøre på separate tråder
    def run_gaussian():
        diff1 = gc.run_comp(org, new)
        print(f"gaussian difference = {diff1}")
        results['diff1'] = diff1

    def run_histogram():
        diff2 = hd.compare_binging_hist_correlation(org, new)
        print(f"histogram difference = {diff2}")
        results['diff2'] = diff2

    # Opprett og start to separate tråder for de to første funksjonene
    results = {}
    threads = [
        threading.Thread(target=run_gaussian),
        threading.Thread(target=run_histogram)
    ]
    for thread in threads:
        thread.start()

    # Kjør den siste funksjonen på hovedtråden og vent på de to andre trådene
    diff3 = sed.get_score(org, new)
    print(f"sobel difference = {diff3}")

    for thread in threads:
        thread.join()

    # Kombiner resultatene og returner gjennomsnittet
    avg_diff = (results['diff1'] + results['diff2'] + diff3) / 3
    return round(avg_diff, 3)


def main():
    compare_img()


if __name__ == '__main__':
    main()
