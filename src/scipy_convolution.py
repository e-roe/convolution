
import cv2
import numpy as np
from scipy import signal

laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

sobel = np.array((
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]), dtype="float")

emboss = np.array((
    [0, -1, 0],
    [0, 0, 0],
    [0, 1, 0]), dtype="int")

prewitt = np.array((
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]), dtype="int")


def convolution(img_in, kernel):
    img = cv2.imread(img_in)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)/255.
    height = img.shape[0]
    width = img.shape[1]
    img_res = np.zeros([height, width, 3], dtype=np.uint8)
    img_res.fill(1)
    filtered = signal.convolve(img, kernel, mode='same')
    cv2.imshow('', filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_file = '../examples/bj1_sml.jpg'
    convolution(img_file, laplacian)
    convolution(img_file, emboss)
    convolution(img_file, prewitt)
    convolution(img_file, sobel)