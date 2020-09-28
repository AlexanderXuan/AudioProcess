import librosa
import numpy as np
import librosa.util as util
from librosa.filters import get_window
import librosa.filters as filters
from scipy import signal
from scipy.io import wavfile
from python_speech_features import mfcc, get_filterbanks


# 读取对应路径的音频文件
def read_audio(audio_path):
    audio_sample, sample_rate = librosa.load(audio_path, sr=None)
    # audio_sample, sample_rate = wavfile.read(audio_path)
    return audio_sample, sample_rate


# 获取窗系数
def hanning(win_len, mode='symmetric'):
    if mode == 'symmetric':
        window = get_window('hann', win_len+2, fftbins=False)
        window = window[1:-1]   # 去掉前后的两个0
    elif mode == 'periodic':
        window = get_window('hann', win_len, fftbins=True)
    else:
        print('Window mode can not be {}'.format(mode))
        raise
    return window


def hamming(win_len, mode='symmetric'):
    if mode == 'symmetric':
        window = get_window('hamm', win_len, fftbins=False)
    elif mode == 'periodic':
        window = get_window('hamm', win_len, fftbins=True)
    else:
        print('Window mode can not be {}'.format(mode))
        raise
    return window


# 对采样点进行分帧
def enframe(x, win, hop_len):
    if isinstance(win, int):
        win_len = win
    elif isinstance(win, np.ndarray):
        win_len = len(win)
    else:
        print('win type is not right.')
        raise
    x_frames = util.frame(x, win_len, hop_len, axis=0)

    if isinstance(win, np.ndarray):
        x_frames = x_frames * win
    return x_frames


# 转换frame到时间
def frame2time(n_frame, frame_len, hop_len, fs):
    frame_time = ((np.arange(1, n_frame + 1) - 1) * hop_len + frame_len / 2) / fs
    return frame_time


# fft
def stftms(x, win, nfft, hop):
    stft_matrix = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=win, window='hann', center=False)
    return stft_matrix.T


# short time power spectrum density (PSD)
def pwelch(x, win_len, hop_len, seg_win, seg_overlap, nfft):
    x_frame = enframe(x, win_len, hop_len)
    pxx = signal.welch(x_frame, window=seg_win, noverlap=seg_overlap, nfft=nfft, axis=-1)
    return pxx


# these two functions change the defalt norm method
def mel_bankm(fs, nfft, mel_num, fmin=0.0, fmax=None):
    # bank = filters.mel(sr=fs, n_fft=nfft, n_mels=mel_num, fmin=fmin, fmax=fmax, norm=None)
    bank = get_filterbanks(nfilt=mel_num, nfft=nfft, samplerate=fs, lowfreq=fmin, highfreq=fmax)
    return bank


def mfcc_m(x, fs, mel_num, win_len, hop_len):
    # mfcc_result = librosa.feature.mfcc(x, sr=fs, n_mfcc=mel_num, win_length=win_len, hop_length=hop_len, n_fft=win_len,
    #                             center=False, norm=None).T
    mfcc_result = mfcc(x, samplerate=fs, winlen=win_len / fs, winstep=hop_len / fs, numcep=mel_num)
    return mfcc_result


def mel_dist(s1, s2, fs, num, win_len, hop_len):
    ccc1 = mfcc_m(s1, fs, num, win_len, hop_len)
    ccc2 = mfcc_m(s2, fs, num, win_len, hop_len)

    dist = np.sqrt(np.sum(np.square(ccc1 - ccc2), axis=-1))
    return dist, ccc1, ccc2


if __name__ == '__main__':
    win = hanning(3)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    enframe(x, win, 2)
    print(type(win))
