import numpy as np
from bm3d import bm3d


def create_kernel(sigma):
    kernel = np.array([[1]])
    kernel = np.real(kernel)
    kernel = kernel / np.sqrt(np.sum(kernel ** 2)) * np.sqrt(sigma)

    return kernel


def bm3d_lib(noisy: np.ndarray, sigma):
    shape = noisy.shape
    kernel = create_kernel(sigma)  # assuming square size

    psd = abs(np.fft.fft2(kernel, shape, axes=(0, 1))) ** 2 * shape[0] * shape[1]

    result = bm3d(noisy, psd)
    result = np.minimum(np.maximum(result, 0), 1)

    return (result * 255).astype(np.uint8)


if __name__ == '__main__':
    import cv2
    img = cv2.imread('test_assets/noise/256_cman_noise.png', 0) / 255
    result = bm3d_lib(img, 30/255)

    while True:
        print('noisy')
        cv2.imshow('Test', img)
        cv2.waitKey(0)

        print('bm3d')
        cv2.imshow('Test', result)
        cv2.waitKey(0)
