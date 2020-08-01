from gaussian_noise import add_gaussian
from controller import Controller
import numpy as np
import cv2


def gaussian_denoising(img):
    kernel = np.array((
        [1 / 16, 1 / 8, 1 / 16],
        [1 / 8, 1 / 4, 1 / 8],
        [1 / 16, 1 / 8, 1 / 16]
    ), dtype='float')

    return cv2.filter2D(img, -1, kernel)


def gaussian_sobel(img, alpha=0.01):
    float_img = img / 255

    x = cv2.Sobel(float_img, cv2.CV_64F, 1, 0)
    y = cv2.Sobel(float_img, cv2.CV_64F, 0, 1)

    edge = np.sqrt(np.square(x) + np.square(y))

    result = gaussian_denoising(float_img) + edge * alpha

    return (result * 255).astype(np.uint8)


def gaussian_canny(img: np.ndarray, t1, t2, alpha=0.1):
    canny = cv2.Canny(img, t1, t2)

    return (gaussian_denoising(img) + img * (canny / 255) * alpha).astype(np.uint8)


def fourier_denoising(img, lowpass_ratio=1.0, highpass_ratio=1.5):
    float_img = img / 255

    lowpass_radius = float_img.shape[0] / 2 * lowpass_ratio
    highpass_radius = float_img.shape[0] / 2 * highpass_ratio
    center = (float_img.shape[0] / 2, float_img.shape[1] / 2)
    f = np.fft.fft2(float_img)
    f = np.fft.fftshift(f)

    for i in range(len(f)):
        for j in range(len(f[i])):
            if highpass_radius ** 2 > (i - center[0]) ** 2 + (j - center[1]) ** 2 > lowpass_radius ** 2:
                f[i, j] = 0

    result = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    return (result * 255).astype(np.uint8)


def fourier_butterworth(img, d0=70/512, d1=280/512, n=1):
    float_img = img / 255
    f = np.fft.fft2(float_img)
    f = np.fft.fftshift(f)
    (w, h) = float_img.shape

    lowpass = np.ones((w, h))
    highpass = np.ones((w, h))

    d0 = d0 * w
    d1 = d1 * w

    for i in range(w):
        for j in range(h):
            d = ((i - w / 2) ** 2 + (j - h / 2) ** 2) ** 0.5
            lowpass[i, j] = 1 / (1 + (np.sqrt(2) - 1) * (d / d0) ** (2 * n))
            highpass[i, j] = 1 - 1 / (1 + (np.sqrt(2) - 1) * (d / d1) ** (2 * n))

    low = f * lowpass
    high = f * highpass
    f = low + high

    result = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    return (result * 255).astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread('./test_assets/base/512_airforce.png', 0)
    print(img.dtype)
    #img = cv2.resize(img, (512, 512))
    img = img / 255
    original = np.copy(img)

    f = np.fft.fft2(img)

    cv2.imshow('Test', img)
    cv2.waitKey(0)

    img, noise = add_gaussian(img, sigma=0.2, clip=False)

    cv2.imshow('Test', img)
    cv2.waitKey(0)

    # fb = fourier_butterworth(img, d0=70, d1=280, n=1)
    gd = gaussian_sobel(img)
    gc = gaussian_canny(img, t1=200, t2=400)

    # print(np.sum(np.square(original - fb)))
    print(np.sum(np.square(original - gd)))

    while True:
        cv2.imshow('Test', noise)
        cv2.waitKey(0)

        cv2.imshow('Test', gc)
        cv2.waitKey(0)

        cv2.imshow('Test', gd)
        cv2.waitKey(0)
