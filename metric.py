import numpy as np
import math
import cProfile


def psnr(img1, img2):
    d = np.array(img1 - img2)
    d = d ** 2
    rmse = np.sum(d) / img1.size

    return 10 * math.log10(255.0 ** 2 / rmse)


if __name__ == '__main__':
    def test(n):
        for _ in range(n):
            print('Test')

    for n in range(10):
        cProfile.run(f'test({n})')
