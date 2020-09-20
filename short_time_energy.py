# author:   xuan
# date:     2020/06/10
# function: print string Hello World!
import numpy as np
from matplotlib import pyplot as plt
from UsingModule.time_frequence_short_time import *


if __name__ == '__main__':
    audio_path = './AudioData/audio/D4_750.wav'
    x, fs = read_audio(audio_path)
    
    win_len = 200
    hop_len = 80
    win = hanning(win_len)
    n_sample = len(x)
    x_frame = enframe(x, win, hop_len)
    n_frame = x_frame.shape[0]
    time = np.arange(0, n_sample, dtype=np.float32) / fs
    en = np.sum(np.square(x_frame), axis=-1)
    plt.subplot(2, 1, 1)
    plt.title("wavform")

