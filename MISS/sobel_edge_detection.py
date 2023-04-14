import cv2
import numpy as np
from MISS.otsus import otsus


def sobel_edge_detection_own(img):
    """Sobel Edge Detection, own implementation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preallocate the matrices with zeros
    I = np.zeros_like(gray)

    # Filter Masks
    F1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    F2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gray = np.float32(gray)

    for i in range(gray.shape[0] - 2):
        for j in range(gray.shape[1] - 2):
            # Gradient operations
            Gx = np.sum(np.multiply(F1, gray[i:i + 3, j:j + 3]))
            Gy = np.sum(np.multiply(F2, gray[i:i + 3, j:j + 3]))

            # Magnitude of vector
            I[i + 1, j + 1] = np.sqrt(Gx ** 2 + Gy ** 2)

    thresh = otsus(I, 256)
    I[I < thresh] = 0
    I[I >= thresh] = 255

    return I


def sobel_edge_detection(image):
    """Sobel Edge Detection, quickest self built implementation"""
    # Define the Sobel kernels for the x and y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to a 1-dimensional array
    flat = gray.flatten()

    # Compute the gradient of the image using the Sobel kernels
    gradient_x = np.abs(np.convolve(flat, sobel_x.flatten(), mode='same'))
    gradient_y = np.abs(np.convolve(flat, sobel_y.flatten(), mode='same'))
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Reshape the gradient image to the original image dimensions
    gradient = gradient.reshape(gray.shape)

    thresh = otsus(image, 256)
    image[image < thresh] = 0
    image[image >= thresh] = 255

    return gradient


def get_score(original, copy):
    """Get the score of the difference between two images that have been passed through the Sobel Edge Detection"""
    # Flatten the images
    o_arr = original.flatten()
    m_arr = copy.flatten()

    # Compare the images
    unlike = np.sum(o_arr != m_arr)

    # Return the score
    return np.round(1 - unlike / np.mean([len(o_arr), len(m_arr)]), 3)


def get_diff(org, new):
    """Get the difference between two images"""
    sobels_org = sobel_edge_detection(org)
    sobels_new = sobel_edge_detection(new)
    diff = get_score(sobels_org, sobels_new)
    return diff