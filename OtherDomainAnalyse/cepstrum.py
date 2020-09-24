# author:   xuan
# date:     2020/09/24
# function: print string Hello World!
import sys
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = r'../AudioData/audio/su1.txt'
    y = np.loadtxt(file)
    fs = 16000
    nfft = 1024
    time = np.arange(0, nfft, dtype=np.float) / fs


    nn = np.arange(0, nfft // 2)
    ff = nn * fs / nfft
    spec = np.log(np.abs(fft(y)))

    cep = ifft(spec)

    mcep = 29
    cep_y = cep[:mcep+1]
    cep_y = np.concatenate((cep_y, np.zeros(nfft - 2 * mcep), cep_y[-1:1:-1]))
    spec_cep = fft(cep_y)

    ft = np.concatenate((np.zeros(mcep), cep[mcep+1:-mcep+1], np.zeros(mcep)))

    spec_ft = fft(ft)

    plt.subplot(4, 1, 1)
    plt.plot(time, y)
    plt.title("waveform")

    plt.subplot(4, 1, 2)
    plt.plot(time, cep)
    plt.title("cepstrum")

    plt.subplot(4, 1, 3)
    plt.plot(ff, spec[:nfft // 2], color='k')
    plt.plot(ff, np.real(spec_cep[:nfft//2]), color='g', linewidth=3)
    plt.title("sepctrum(black) and Channel impulse response spectrum(green)")


    plt.subplot(4, 1, 4)
    plt.plot(ff, np.real(spec_ft[:nfft//2]))
    plt.title("Glottis excited pulse spectrum")

    plt.show()

    print("Done.")
