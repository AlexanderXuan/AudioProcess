# author:   xuan
# date:     2020/9/27
# function: print string Done.
import numpy as np
import matplotlib.pyplot as plt
from UsingModule.time_frequence_short_time import read_audio, mel_dist


if __name__ == '__main__':
    audio_path1 = '../AudioData/audio/s1.wav'
    audio_path2 = '../AudioData/audio/s2.wav'
    audio_path3 = '../AudioData/audio/a1.wav'
    x1, fs = read_audio(audio_path1)
    x2, _ = read_audio(audio_path2)
    x3, _ = read_audio(audio_path3)

    win_len = 200
    hop_len = 80
    x1 = x1 / np.max(np.abs(x1))
    x2 = x2 / np.max(np.abs(x2))
    x3 = x3 / np.max(np.abs(x3))

    dist, ccep1, ccep2 = mel_dist(x1, x2, fs, 16, win_len, hop_len)
    plt.subplot(2, 1, 1)
    plt.scatter(ccep1[2], ccep2[2], marker='+')
    plt.legend('3 frame')
    plt.scatter(ccep1[6], ccep2[7], marker='x')
    plt.legend('7 frame')
    plt.scatter(ccep1[11], ccep2[12], marker='^')
    plt.legend('12 frame')
    plt.scatter(ccep1[15], ccep2[16], marker='h')
    plt.legend('16 frame')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis([-40, 40, -40, 40])
    plt.plot([-40, 40], [-40, 40], color='k', linestyle='-')
    plt.title("/i1/ and /i2/ mfcc distance")

    dist2, ccep1, ccep3 = mel_dist(x1, x3, fs, 16, win_len, hop_len)
    plt.subplot(2, 1, 2)
    plt.scatter(ccep1[2], ccep3[2], marker='+')
    plt.legend('3 frame')
    plt.scatter(ccep1[6], ccep3[7], marker='x')
    plt.legend('7 frame')
    plt.scatter(ccep1[11], ccep3[12], marker='^')
    plt.legend('12 frame')
    plt.scatter(ccep1[15], ccep3[16], marker='h')
    plt.legend('16 frame')
    plt.xlabel('x1')
    plt.ylabel('x3')
    plt.axis([-40, 40, -40, 40])
    plt.plot([-40, 40], [-40, 40], color='k', linestyle='-')
    plt.title("/i1/ and /a1/ mfcc distance")
    plt.show()
    print("Done.")
