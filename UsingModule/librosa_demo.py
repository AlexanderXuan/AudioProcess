import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

audio_dir = "../AudioData/audio"
noise_dir = "../AudioData/noise"
audio_path = "../AudioData/audio/D4_750.wav"
noise_path = "../AudioData/noise/Pink Noise.wav"

# ----------------------------------- 1. 处理音频 -------------------------------------

# 直接读取
audio_sample, sample_rate = librosa.load(audio_path)
# 如果需要重采样则设置sr参数，如果需要转换为单声道则设置mono参数

# 只获得音频采样率
sample_rate = librosa.get_samplerate(audio_path)

# 转换为单声道功能
audio_sample = librosa.to_mono(audio_sample)

# 重采样功能
target_sample_rate = 8000
audio_sample = librosa.resample(audio_sample, orig_sr=sample_rate, target_sr=target_sample_rate)

# 获得音频时间
audio_duration = librosa.get_duration(audio_sample, sr=target_sample_rate)
# 这个函数同样可以计算stft之后的音频时间，将在之后进行演示

# 获得信号的自相关序列，这里的实现方式是先算功率谱，然后IFFT得到自相关序列
auto_correlate = librosa.autocorrelate(audio_sample)

# 获得LPC系数，使用Burg方法，order是阶数
lpc_co = librosa.lpc(audio_sample, order=16)

# 获得过零点
zero_crossing = librosa.zero_crossings(audio_sample)
# 由这个应该可以用来计算过零率

# ----------------------------------- 1. 频谱变换 -------------------------------------
# 短时傅立叶变换   分帧加窗等操作可以进入到这个函数中找到
map_f_t = librosa.stft(audio_sample, n_fft=256, hop_length=64, win_length=256, window='hann', center=True)
# 获得幅度
amp_f_t = np.abs(map_f_t)
# 获得相角
ang_f_t = np.angle(map_f_t)
# 获得功率谱
pow_f_t = amp_f_t ** 2

# 反短时傅立叶变换
inversed_sample = librosa.istft(map_f_t, hop_length=64, win_length=256, window='hann')

# 画频谱图
plt.subplot(2, 1, 1)
D = librosa.amplitude_to_db(amp_f_t, ref=np.max)
librosa.display.specshow(D, sr=target_sample_rate, hop_length=64, x_axis='frames', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency amplitude spectrogram')
# 画波形图
plt.subplot(2, 1, 2)
librosa.display.waveplot(audio_sample, sr=target_sample_rate)
plt.title('audio waveform')
plt.show()


