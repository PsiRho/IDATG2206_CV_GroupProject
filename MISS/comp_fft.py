import numpy as np


def compare_fft(original_image, degraded_image):
    # Compute the FFT of both images
    original_fft = np.fft.fft2(original_image)
    degraded_fft = np.fft.fft2(degraded_image)

    # Compute the magnitude of the FFT
    original_mag = np.abs(original_fft)
    degraded_mag = np.abs(degraded_fft)

    # Normalize the magnitudes to [0, 1]
    original_mag /= original_mag.max()
    degraded_mag /= degraded_mag.max()

    # Compute the absolute difference between the magnitudes
    diff = np.abs(original_mag - degraded_mag)

    # Compute the score as the mean of the difference
    score = 1 - diff.mean()

    return round(score, 3)
