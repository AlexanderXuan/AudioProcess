# author:   xuan
# date:     9/21/20
# file name: short_time_freq
# function: print string Done!
import numpy as np
from matplotlib import pyplot as plt
import librosa.display
import librosa
from UsingModule.time_frequence_short_time import *


if __name__ == '__main__':
    audio_path = '../AudioData/audio/D4_750.wav'
    x, fs = read_audio(audio_path)

    win_len = 200
    hop_len = 80
    n_sample = len(x)
    stft_result = stftms(x, win_len, win_len, hop_len)
    n_frame = stft_result.shape[0]
    frame_time = frame2time(n_frame, win_len, hop_len, fs)

    amp_f_t = np.abs(stft_result)
    # D = amp_f_t

    # 画频谱图
    plt.subplot(2, 1, 1)
    D = librosa.amplitude_to_db(amp_f_t, ref=np.max)
    librosa.display.specshow(D.T, sr=fs, hop_length=hop_len, x_axis='frames', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency amplitude spectrogram')
    # 画波形图
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x, sr=fs)
    plt.title('audio waveform')
    plt.show()
    print("Done.")
