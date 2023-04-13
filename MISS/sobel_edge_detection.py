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

    # Convert the gradient image to a 8-bit unsigned integer
    # TODO: denne endrer score en del. Prøv å comment ut og kjør igjen.
    # gradient = np.uint8(gradient)

    return gradient


def sobel_edge_detection_inbuilt(img):
    """Sobel Edge Detection, using OpenCV inbuilt function"""
    gray = cv2.cv2tColor(img, cv2.COLOR_BGR2GRAY)

    # Preallocate the matrices with zeros
    I = np.zeros_like(gray)

    grad_x = cv2.Sobel(gray, cv2.cv2_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.cv2_64F, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    I = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    thresh = otsus(I, 256)
    I[I < thresh] = 0
    I[I >= thresh] = 255

    return I


def get_score(original, copy):
    """Get the score of the difference between two images"""
    # Flatten the images
    o_arr = original.flatten()
    m_arr = copy.flatten()
    unlike = 0

    # Compare the images
    for i in range(1, len(o_arr)):
        if o_arr[i] != m_arr[i]:
            unlike += 1

    # Return the score
    return np.round(1 - unlike / ((len(o_arr) + len(m_arr)) / 2), 3)


def get_diff(org, new):
    """Get the difference between two images"""
    sobels_org = sobel_edge_detection(org)
    sobels_new = sobel_edge_detection(new)
    diff = get_score(sobels_org, sobels_new)
    return diff


"""
img1 = cv2.imread('../CIDIQ_Dataset/Images/Original/final01.bmp')
img2 = cv2.imread('../CIDIQ_Dataset/Images/Reproduction/1_JPEG2000_Compression/final01_d1_l1.bmp')  #
og = sobel_edge_detection(img1)
new = sobel_edge_detection(img2)  #
print(get_score(og, new))
"""
