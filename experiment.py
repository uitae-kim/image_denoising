from gaussian_noise import add_gaussian
from classical_denoising import gaussian_denoising, gaussian_sobel, gaussian_canny, fourier_denoising, fourier_butterworth
from wiener import wiener_filter
from pre_made.BM3D import bm3d
import numpy as np
import cv2
import cProfile
from metric import psnr
import os
import itertools


# =====================================
# data structure
class Image:
    def __init__(self, img, fname):
        self.image = img
        self.filename = fname.split('.')[0].split('_')[1]
# =====================================


# =====================================
# settings
def load():
    size = [64, 128, 256, 512]
    path = './test_assets/base'
    fnames = os.listdir(path)

    result = {}
    for s in size:
        result[s] = []

    for fname in fnames:
        if '.png' in fname:
            for s in size:
                if str(s) in fname:
                    result[s].append(
                        Image(cv2.imread(path + '/' + fname, 0), fname)
                    )

    return result


def awgn(images, sigma):
    result = []
    for img in images:
        noisy, noise = add_gaussian(img / 255, sigma=sigma / 255)
        result.append((noisy * 255).astype(np.uint8))

    return result
# =====================================


# =====================================
# experiments
def classical_exp(images, noisy_list, show=False):
    assert(len(images) == len(noisy_list))

    result = []

    for i in range(len(images)):
        image = images[i]
        noisy = noisy_list[i]
        gd = gaussian_denoising(noisy)
        gs = gaussian_sobel(noisy)
        median = np.median(noisy)
        lower = int(max(0, median * 0.7))
        upper = int(min(255, median * 1.3))
        gc = gaussian_canny(noisy, t1=lower, t2=upper)
        fd = fourier_denoising(noisy)
        fb = fourier_butterworth(noisy)

        result_by_image = [
            psnr(image, gd),
            psnr(image, gs),
            psnr(image, gc),
            psnr(image, fd),
            psnr(image, fb)
        ]
        result.append(result_by_image)

        if show:
            cv2.imshow('Test', gd)
            cv2.waitKey(0)
            cv2.imshow('Test', gs)
            cv2.waitKey(0)
            cv2.imshow('Test', gc)
            cv2.waitKey(0)
            cv2.imshow('Test', fd)
            cv2.waitKey(0)
            cv2.imshow('Test', fb)
            cv2.waitKey(0)

    return result


def improved_fourier(images, noisy_list, show=False):
    assert (len(images) == len(noisy_list))

    result = []

    for i in range(len(images)):
        result_by_image = []
        image = images[i]
        noisy = noisy_list[i]
        w_psnr = []
        for k in range(1, 10):
            wiener = wiener_filter(noisy, k)
            w_psnr.append(psnr(image, wiener))
            if show:
                cv2.imshow('Test', wiener)
                cv2.waitKey(0)
        result_by_image.append(max(w_psnr))

        bm3d_psnr = []

        block_size = [4, 8, 16, 32]
        beta_kaiser = [1.0, 2.0, 4.0, 8.0]
        first_thres_bm = [1000, 2000, 3000, 4000]
        second_thres_bm = [1000, 2000, 3000, 4000]
        sigma_list = [10, 25, 50]

        chain = itertools.chain(itertools.product(block_size, beta_kaiser, first_thres_bm, second_thres_bm, sigma_list))

        for c in chain:
            sigma = c[4]
            first_params = {
                'block_size': c[0],
                'block_step': 3,
                'beta_kaiser': c[1],
                'search_step': 3,
                'threshold_bm': c[2],
                'max_matches': 16,
                'window_size': 32,
                'threshold_3d': 2.7 * sigma
            }

            second_params = {
                'block_size': c[0],
                'block_step': 3,
                'search_step': 3,
                'threshold_bm': c[3],
                'max_matches': 32,
                'window_size': 32
            }

            bm3d_img = bm3d(noisy / 255, first_params, second_params, sigma)
            bm3d_psnr.append(psnr(image, bm3d_img))
        result_by_image.append(max(bm3d_psnr))

        result.append(result_by_image)

    return result
# =====================================


if __name__ == '__main__':
    image_dict = load()
    sigma_list = [15, 25, 50]

    for size in [64]:
        images = list(map(lambda img: img.image, image_dict[size]))
        for sigma in sigma_list:
            noisy_list = awgn(images, sigma)
            classical_results = classical_exp(images, noisy_list)
            extended_results = improved_fourier(images, noisy_list)

            results = []
            for i in range(len(images)):
                results.append(classical_results[i] + extended_results[i])

            with open(f'./experiments/{size:03d}_{sigma}.csv', 'w') as f:
                f.write('image,gaussian,sobel,canny,fourier,butterworth,wiener,bm3d\n')

                for i, result in enumerate(results):
                    fname = image_dict[size][i].filename
                    result_str = ','.join(map(str, result))
                    f.write(f'{fname},{result_str}\n')

