import librosa
import numpy as np
import librosa.util as util
from librosa.filters import get_window


# 读取对应路径的音频文件
def read_audio(audio_path):
    audio_sample, sample_rate = librosa.load(audio_path)
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
def frame2time():
    pass

if __name__ == '__main__':
    win = hanning(3)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    enframe(x, win, 2)
    print(type(win))
