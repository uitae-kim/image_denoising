import numpy as np
import cv2


def gaussian_kernel():
    return np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 24, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 24, 16, 4],
        [1, 4, 7, 4, 1],
    ]) / 257


def wiener_filter(img, K):
    kernel = gaussian_kernel()
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.fft.fftshift(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    f = f * kernel
    f = np.fft.ifftshift(f)
    recon = np.abs(np.fft.ifft2(f))
    return recon / np.sum(recon) * np.sum(img)


if __name__ == '__main__':
    import time

    t = time.time()
    from gaussian_noise import add_gaussian
    img = cv2.imread('../test_assets/base/512_lenna.png', 0)
    img = img / 255
    noisy, noise = add_gaussian(img, sigma=0.2, clip=False)

    filtered = wiener_filter(noisy, K=10)

    from classical_denoising import gaussian_denoising, fourier_butterworth
    gd = gaussian_denoising(noisy)
    fb = fourier_butterworth(noisy, d0=75, d1=250)

    print(time.time() - t)

    while True:
        cv2.imshow('Test', img)
        cv2.waitKey(0)

        cv2.imshow('Test', noisy)
        cv2.waitKey(0)

        cv2.imshow('Test', filtered)
        cv2.waitKey(0)

        cv2.imshow('Test', gd)
        cv2.waitKey(0)

        cv2.imshow('Test', fb)
        cv2.waitKey(0)
