import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

blur = np.array((
    [1., 1., 1.],
    [1, 1, 1.],
    [1., 1., 1.]), dtype="float")

Sobel_x = np.array((
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]), dtype="float")

Sobel_y = np.array((
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]), dtype="float")

prewitt = np.array((
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]), dtype="float")

laplacian = np.array((
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]), dtype="float")

emboss = np.array((
    [-1, -1, 0],
    [-1, 0, 1],
    [0, 1, 1]), dtype="float")

emboss2 = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="float")

blur_5 = np.array((
    [1/9, 1/9, 1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9, 1/9, 1/9]), dtype="float")

result_path = '../results'


def plot_and_save(grayscale_image, filtered_image, filter_name):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    w = grayscale_image.shape[1]
    h = grayscale_image.shape[0]
    fig2 = plt.figure(2, figsize=(15,7))
    ax1, ax2 = fig2.add_subplot(121), fig2.add_subplot(122)
    ax1.imshow(grayscale_image, cmap=plt.get_cmap('gray'))
    ax1.set(title='Original Image')
    ax2.set(title=filter_name)
    ax2.imshow(filtered_image, cmap=plt.get_cmap('gray'))

    plt.figure(1, figsize=(w * px, h * px))
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(result_path, f'{filter_name}.jpg'),  bbox_inches='tight', pad_inches=0)

    plt.imshow(grayscale_image, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(result_path, f'original_{filter_name}.jpg'), bbox_inches='tight', pad_inches=0)


def emboss_filter(img_in, kernel, factor, bias, filter_name):
    print(f'Applying {filter_name}')
    grayscale_image = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
    height = grayscale_image.shape[0]
    width = grayscale_image.shape[1]
    filtered_image = np.zeros([height, width, 3], dtype=np.uint8)
    height_k = kernel.shape[0]
    width_k = kernel.shape[1]
    init_x = kernel.shape[1] // 2
    init_y = kernel.shape[0] // 2
    for x in range(init_x, width - init_x):
        for y in range(init_y, height - init_y):
            resulting_pixel = 0
            for xk in range(width_k):
                for yk in range(height_k):
                    resulting_pixel += kernel[yk, xk] * grayscale_image[y + (yk-init_y), x + (xk-init_x)]
            filtered_image[y, x] = int((resulting_pixel / factor) + bias)

    plt.figure(1, figsize=(10, 10))
    plt.imshow(kernel, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(result_path, f'Kernel_{filter_name}.jpg'), bbox_inches='tight', pad_inches=0)
    plt.close()

    plot_and_save(grayscale_image, filtered_image, filter_name)


def apply_filter(input_image, kernel_x, kernel_y=None, filter_name=''):
    print(f'Applying filter {filter_name}')
    grayscale_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    rows, columns = np.shape(grayscale_image)
    filtered_image = np.zeros(shape=(rows, columns))
    kernel_dim = kernel_x.shape[0]
    for i in range(rows - (kernel_dim-1)):
        for j in range(columns - (kernel_dim-1)):
            gy = 0
            gx = abs(np.sum(np.multiply(kernel_x, grayscale_image[i:i + kernel_dim, j:j + kernel_dim])))
            if kernel_y is not None:
                gy = np.sum(np.multiply(kernel_y, grayscale_image[i:i + kernel_dim, j:j + kernel_dim]))  # y direction
            filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

    plt.figure(1, figsize=(10, 10))
    plt.imshow(kernel_x, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(result_path, f'Kernel_{filter_name}.jpg'), bbox_inches='tight', pad_inches=0)
    plt.close()

    plot_and_save(grayscale_image, filtered_image, filter_name)


if __name__ == '__main__':
    image_file = '../examples/bj1_sml.jpg'
    os.makedirs(result_path, exist_ok=True)
    emboss_filter(image_file, emboss, 9, 128, filter_name='Emboss')
    emboss_filter(image_file, emboss2, 9, 128, filter_name='Emboss2')
    apply_filter(image_file, blur, filter_name='Blur')
    apply_filter(image_file, blur_5, filter_name='Blur_5')
    apply_filter(image_file, laplacian, filter_name='Laplacian')
    apply_filter(image_file, Sobel_x, filter_name='SobelX')
    apply_filter(image_file, Sobel_y, filter_name='SobelY')
    apply_filter(image_file, Sobel_x, Sobel_y, filter_name='Sobel')