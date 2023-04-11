import cv2 as cv
import numpy as np
from MISS.otsus import otsus

org = cv.imread('../CIDIQ_Dataset/Images/Original/final07.bmp')
comp = cv.imread('../CIDIQ_Dataset/Images/Reproduction/3_Poisson_Noise/final02_d3_l1.bmp')

def sobel_edge_detection(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Preallocate the matrices with zeros
    I = np.zeros_like(gray)

    # Filter Masks
    F1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    F2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    A = np.float32(gray)

    for i in range(gray.shape[0]-2):
        for j in range(gray.shape[1]-2):
            # Gradient operations
            Gx = np.sum(np.multiply(F1, gray[i:i+3, j:j+3]))
            Gy = np.sum(np.multiply(F2, gray[i:i+3, j:j+3]))

            # Magnitude of vector
            I[i+1, j+1] = np.sqrt(Gx**2 + Gy**2)

    cv.imshow('Original', img)
    cv.waitKey(0)

    #I = np.uint8(I)
    cv.imshow('Filtered Image', I)
    cv.waitKey(0)

    thresh = otsus(I, 256)
    I[I < thresh] = 0
    I[I > thresh] = 255

    cv.imshow('Edge detected Image', I)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return I

original = sobel_edge_detection(org)
manipulated = sobel_edge_detection(comp)

def get_score(original, copy):
    o_arr = original.flatten()
    m_arr = copy.flatten()
    unlike = 0

    for i in range(1, len(o_arr)):
        if o_arr[i] != m_arr[i]:
            unlike += 1

    print(1 - unlike/((len(o_arr)+len(m_arr))/2))

    return 1 - unlike/((len(o_arr)+len(m_arr))/2)
get_score(original, manipulated)
