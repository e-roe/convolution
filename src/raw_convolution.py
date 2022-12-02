
import cv2
import numpy as np

blur_3x3 = np.array((
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]), dtype="int")

blur_5x5 = np.array((
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]), dtype="int")


def convolution(img_in, kernel):
    img = cv2.imread(img_in)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) / 255.
    height = img.shape[0]
    width = img.shape[1]
    img_res = np.zeros([height, width, 3], dtype=np.uint8)
    img_res.fill(1)
    height_k = kernel.shape[0]
    width_k = kernel.shape[1]
    factor = height_k * width_k
    init_x = kernel.shape[1] // 2
    init_y = kernel.shape[0] // 2
    for x in range(init_x, width - init_x):
        for y in range(init_y, height - init_y):
            resulting_pixel = 0
            for xk in range(width_k):
                for yk in range(height_k):
                    resulting_pixel += kernel[yk, xk] * img[y + (yk-init_y), x + (xk-init_x)]
            img_res[y, x] = int((resulting_pixel / factor) * 255)

    cv2.imshow('', img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_file = '../examples/a.jpg'
    convolution(img_file, blur_3x3)
    convolution(img_file, blur_5x5)