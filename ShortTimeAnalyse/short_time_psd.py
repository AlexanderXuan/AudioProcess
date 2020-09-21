# author:   xuan
# date:     9/21/20
# file name: short_time_psd
# function: print string Done!
from matplotlib import pyplot as plt
import librosa.display
import librosa
from UsingModule.time_frequence_short_time import *


if __name__ == '__main__':
    audio_path = '../AudioData/audio/D4_750.wav'
    x, fs = read_audio(audio_path)

    win_len = 240
    hop_len = 80
    seg_win = hanning(200)# .reshape((1, -1))
    seg_overlap = 195
    nfft = 200
    f, pxx = pwelch(x, win_len, hop_len, seg_win, seg_overlap, nfft)
    n_frame = pxx.shape[0]
    frame_time = frame2time(n_frame, nfft, hop_len, fs)
    freq = np.arange(0, nfft // 2 + 1, dtype=np.float) * fs / nfft
    librosa.display.specshow(pxx.T, sr=fs, hop_length=hop_len, x_axis='frames', y_axis='linear')
    plt.title('short time psd')
    plt.show()
    print("Done.")
