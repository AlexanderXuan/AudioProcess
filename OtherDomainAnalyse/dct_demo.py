# author:   xuan
# date:     2020/09/26
# function: print string Hello World!
import sys
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt


if __name__ == '__main__':
    f = 50
    fs = 1000
    n_sampels = 1000
    n = np.arange(0, n_sampels)
    xn = np.cos(2 * np.pi * f * n / fs)
    y = dct(xn)
    num = np.where(np.abs(y) < 5)
    y[num] = 0

    zn = idct(y)

    plt.subplot(2, 1, 1)
    plt.plot(n, xn)
    plt.title("xn")

    plt.subplot(2, 1, 2)
    plt.plot(n, zn)
    plt.title("zn")

    plt.show()

    print("Done.")
