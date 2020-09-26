# author:   xuan
# date:     2020/9/26
# function: print string Done.
import numpy as np
import matplotlib.pyplot as plt
from UsingModule.time_frequence_short_time import mel_bankm


if __name__ == '__main__':
    fs = 8000
    nfft = 256
    num_mel = 24
    fmin = 0
    fmax = 0.5 * fs
    bank = mel_bankm(fs, nfft, num_mel, fmin, fmax)
    bank = bank / np.max(bank)

    df = fs / nfft
    ff = np.arange(0, nfft // 2 + 1) * df
    for k in range(num_mel):
        plt.plot(ff, bank[k])
    plt.grid()
    plt.title("mel filter banks freq response")
    plt.show()
    print("Done.")
