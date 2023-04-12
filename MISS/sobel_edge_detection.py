import cv2 as cv
import numpy as np
from PIL import Image
from MISS.otsus import otsus

def sobel_edge_detection_own(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Preallocate the matrices with zeros
    I = np.zeros_like(gray)

    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    I = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    thresh = otsus(I, 256)
    I[I < thresh] = 0
    I[I >= thresh] = 255

    return I

def sobel_edge_detection_inbuilt(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Preallocate the matrices with zeros
    I = np.zeros_like(gray)

    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    I = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    thresh = otsus(I, 256)
    I[I < thresh] = 0
    I[I >= thresh] = 255

    return I


def get_score(original, copy):
    o_arr = original.flatten()
    m_arr = copy.flatten()
    unlike = 0

    for i in range(1, len(o_arr)):
        if o_arr[i] != m_arr[i]:
            unlike += 1

    return np.round(1 - unlike/((len(o_arr)+len(m_arr))/2), 3)


def get_diff(org, new):
    sobels_org = sobel_edge_detection_inbuilt(org)
    sobels_new = sobel_edge_detection_inbuilt(new)
    diff = get_score(sobels_org, sobels_new)
    return diff