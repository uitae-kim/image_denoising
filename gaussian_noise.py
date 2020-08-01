import numpy as np
import cv2


def add_gaussian(img: np.ndarray, mean=0, sigma=0.3, clip=False):
    noise = np.random.normal(mean, sigma, img.shape)
    if clip:
        noise = np.clip(noise, 0, 1)

    return noise + img, noise


if __name__ == '__main__':
    img = cv2.imread('5544.jpg', 0)
    img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    img = img / 255

    cv2.imshow('Test', img)
    cv2.waitKey(0)
    cv2.imshow('Test', add_gaussian(img, sigma=0.2, clip=False))
    cv2.waitKey(0)

    kernel = np.array((
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
    ), dtype='float')

    img = cv2.filter2D(img, -1, kernel)
    cv2.imshow('Test', img)
    cv2.waitKey(0)
