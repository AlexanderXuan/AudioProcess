import librosa
import numpy as np
import librosa.util as util
from librosa.filters import get_window
import scipy


def butter_filter(data, sr, fp, fs, rp, rs):
    """
    :arg
    data: the data need to filter, one dimention
    sr: sample rate
    fp: pass frequency
    fs: stop frequency
    rp: pass band ripple
    rs: stop band ripple
    :return
    filtered_data: data after filtered
    """
    sr2 = sr // 2
    wp = fp / sr2   # 通带频率归一化
    ws = fs / sr2   # 阻带频率归一化
    n, wn = scipy.signal.buttord(wp, ws, rp, rs)
    b, a = scipy.signal.butter(n, wn)
    return scipy.signal.lfilter(b, a, data)


def cheby1_filter(data, sr, fp, fs, rp, rs):
    """
    :arg
    data: the data need to filter, one dimention
    sr: sample rate
    fp: pass frequency
    fs: stop frequency
    rp: pass band ripple
    rs: stop band ripple
    :return
    filtered_data: data after filtered
    """
    sr2 = sr // 2
    wp = fp / sr2   # 通带频率归一化
    ws = fs / sr2   # 阻带频率归一化
    n, wn = scipy.signal.cheb1ord(wp, ws, rp, rs)
    b, a = scipy.signal.cheby1(n, rp, wn)
    return scipy.signal.lfilter(b, a, data)


def cheby2_filter(data, sr, fp, fs, rp, rs):
    """
    :arg
    data: the data need to filter, one dimention
    sr: sample rate
    fp: pass frequency
    fs: stop frequency
    rp: pass band ripple
    rs: stop band ripple
    :return
    filtered_data: data after filtered
    """
    sr2 = sr // 2
    wp = fp / sr2   # 通带频率归一化
    ws = fs / sr2   # 阻带频率归一化
    n, wn = scipy.signal.cheb2ord(wp, ws, rp, rs)
    b, a = scipy.signal.cheby2(n, rs, wn)
    return scipy.signal.lfilter(b, a, data)


def ellips_filter(data, sr, fp, fs, rp, rs):
    """
    :arg
    data: the data need to filter, one dimention
    sr: sample rate
    fp: pass frequency
    fs: stop frequency
    rp: pass band ripple
    rs: stop band ripple
    :return
    filtered_data: data after filtered
    """
    sr2 = sr // 2
    wp = fp / sr2   # 通带频率归一化
    ws = fs / sr2   # 阻带频率归一化
    n, wn = scipy.signal.ellipord(wp, ws, rp, rs)
    b, a = scipy.signal.ellip(n, rp, rs, wn)
    return scipy.signal.lfilter(b, a, data)

