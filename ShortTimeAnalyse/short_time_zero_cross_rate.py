# author:   xuan
# date:     2020/09/20
# file name: short_time_zero_cross_rate.py
# function: print string Done!
import numpy as np
from matplotlib import pyplot as plt
from UsingModule.time_frequence_short_time import *


if __name__ == '__main__':
    audio_path = '../AudioData/audio/D4_750.wav'
    x, fs = read_audio(audio_path)

    win_len = 200
    hop_len = 80
    win = hanning(win_len)
    n_sample = len(x)
    x_frame = enframe(x, win_len, hop_len)
    n_frame = x_frame.shape[0]
    zcr1 = np.sum(x_frame[:, :-1] * x_frame[:, 1:] < 0, axis=-1)
    time = np.arange(0, n_sample, dtype=np.float32) / fs
    frame_time = frame2time(n_frame, win_len, hop_len, fs)
    plt.subplot(2, 1, 1)
    plt.title("wavform")
    plt.plot(time, x)
    plt.subplot(2, 1, 2)
    plt.title("short time zero cross rate")
    plt.plot(frame_time, zcr1)
    plt.show()
    print("Done.")
