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
    float_img = img / 255

    kernel = gaussian_kernel()
    f = np.fft.fft2(float_img)
    f = np.fft.fftshift(f)

    kernel = np.fft.fft2(kernel, s=float_img.shape)
    kernel = np.fft.fftshift(kernel)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    f = f * kernel
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

    result = recon / np.sum(recon) * np.sum(float_img)

    return (result * 255).astype(np.uint8)


if __name__ == '__main__':
    from gaussian_noise import add_gaussian
    img = cv2.imread('./test_assets/base/512_lenna.png', 0)
    img = img / 255
    noisy, noise = add_gaussian(img, sigma=30/255)

    result = wiener_filter(noisy, K=1)

    from classical_denoising import gaussian_denoising, fourier_denoising, fourier_butterworth
    gd = gaussian_denoising(noisy)
    fd = fourier_denoising(noisy)
    fb = fourier_butterworth(noisy, d0=70, d1=280, n=1)

    cv2.imshow('Test', img)
    cv2.waitKey(0)
    cv2.imshow('Test', noisy)
    cv2.waitKey(0)

    together = cv2.vconcat([cv2.hconcat([result, gd]), cv2.hconcat([fd, fb])])

    cv2.imshow('Test', together)
    cv2.waitKey(0)
