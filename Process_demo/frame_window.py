import librosa
import numpy as np
import librosa.util as util
from librosa.filters import get_window


audio_path = "../AudioData/audio/D4_750.wav"
noise_path = "../AudioData/noise/Pink Noise.wav"
# 读取音频文件
y, sr = librosa.load(audio_path)

# 对音频文件进行分帧
win_len = n_fft = 200
hop_length = 80
# Pad the time series so that frames are centered
y = np.pad(y, int(n_fft // 2), mode='reflect')
# Window the time series.
y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length, axis=0)

# 获得窗系数
fft_window = get_window('hamm', 10, fftbins=False)
# fft_window = fft_window[1:-1]
print(fft_window)
fft_window = get_window('hamm', 10, fftbins=True)
print(fft_window)
# Pad the window out to n_fft size
fft_window = util.pad_center(fft_window, n_fft)
# Reshape so that the window can be broadcast
fft_window = fft_window.reshape((-1, 1))

#