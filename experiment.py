from gaussian_noise import add_gaussian
from classical_denoising import gaussian_denoising, gaussian_sobel, gaussian_canny, fourier_denoising, fourier_butterworth
from wiener import wiener_filter
from pre_made.BM3D import bm3d
from bm3d_lib import bm3d_lib
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
        self.size = fname.split('.')[0].split('_')[0]
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
    result_images = []

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
        result_images.append([gd, gs, gc, fd, fb])

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

    return result, result_images


def improved_fourier(images, noisy_list, sigma_list, show=False):
    assert (len(images) == len(noisy_list))

    result = []
    result_images = []

    for i in range(len(images)):
        result_by_image = []
        image = images[i]
        noisy = noisy_list[i]
        w_psnr = 0
        w_image = None
        for k in range(1, 10):
            wiener = wiener_filter(noisy, k)
            temp = psnr(image, wiener)
            if temp > w_psnr:
                w_psnr = temp
                w_image = wiener
        result_by_image.append(w_psnr)

        bm3d_psnr = 0
        b_img = None

        for sigma in sigma_list:
            bm3d_img = bm3d_lib(noisy / 255, sigma / 255)
            temp = psnr(image, bm3d_img)
            if show:
                cv2.imshow('Test', bm3d_img)
                cv2.waitKey(0)
            if temp > bm3d_psnr:
                bm3d_psnr = temp
                b_img = bm3d_img

        result_by_image.append(bm3d_psnr)

        result.append(result_by_image)
        result_images.append([w_image, b_img])

    return result, result_images
# =====================================


if __name__ == '__main__':
    image_dict = load()
    sigma_list = [15, 25, 50]

    if not os.path.exists('result'):
        os.mkdir('result')

    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    header = 'gaussian,sobel,canny,fourier,butterworth,wiener,bm3d'
    header_list = header.split(',')

    for size in image_dict:
        images = list(map(lambda img: img.image, image_dict[size]))
        for sigma in sigma_list:
            noisy_list = awgn(images, sigma)
            classical_results, classical_imgs = classical_exp(images, noisy_list)
            extended_results, extended_imgs = improved_fourier(images, noisy_list, sigma_list)

            results = []
            result_images = []
            for i in range(len(images)):
                results.append(classical_results[i] + extended_results[i])
                result_images.append(classical_imgs[i] + extended_imgs[i])

            with open(f'./experiments/{size:03d}_{sigma}.csv', 'w') as f:
                f.write('image,gaussian,sobel,canny,fourier,butterworth,wiener,bm3d\n')

                for i, result in enumerate(results):
                    fname = image_dict[size][i].filename
                    result_str = ','.join(map(str, result))
                    f.write(f'{fname},{result_str}\n')

                    for j, img in enumerate(result_images[i]):
                        cv2.imwrite(f'./result/{size:03d}_{sigma}_{header_list[j]}_{fname}.png', img)
