from __future__ import print_function

import numpy as np
import pandas as pd
from numpy.fft import fftpack


def calc_freq(time, sample_rate):
    N = len(time)
    Fs = 1.0 / (max(time) - min(time))
    Fn = 0.5 * sample_rate
    freq = np.array([-Fn + i * Fs for i in range(N)])
    return freq

def dofft(iq):
    N = len(iq)
    iq_fft = np.fft.fftshift(fftpack.fft(iq))  # fft and shift axis
    # iq_fft = 20 * np.log10(abs((iq_fft + 1e-15) / N))  # convert to decibels, adjust power

    # adding 1e-15 (-300 dB) to protect against value errors if an item in iq_fft is 0
    return iq_fft

def moving_avg(data, window_size=48, scale=1.0):
    assert len(data) >= window_size, 'the length of data should not be smaller then nwindow'
    sum_list = []
    sum = np.sum(data[:window_size])
    sum_list.append(sum*scale)
    for i in range(1, len(data)):
        # print(data[i-1])
        if i + window_size - 1 < len(data):
            sum += data[i + window_size -1]
            sum -= data[i - 1]
        else:
            sum -= data[i - 1]
        sum_list.append(sum*scale)
    return np.array(sum_list)

def convert_seq2start_len(data_indices):
    if len(data_indices) == 0:
        return pd.DataFrame({'start_index': [0], 'length': [0]})
    start_list = []
    length_list = []
    start = data_indices[0]
    start_index = 0
    seq_len = 1
    for i in range(1, len(data_indices)):
        index = data_indices[i]
        if i - start_index == index - start:
            seq_len += 1
        else:
            start_list.append(start)
            length_list.append(seq_len)
            start = index
            start_index = i
            seq_len = 1
    start_list.append(start)
    length_list.append(seq_len)

    return pd.DataFrame({'start_index': start_list, 'length': length_list})


def acf_norm(data, ndelay=16, nwindow=48):
    assert len(data) >= nwindow, 'the length of data should not be smaller then nwindow'
    a_list = []
    p_list = []
    data_padding = np.concatenate([data, np.zeros(ndelay)]).astype(np.complex64)
    padding_data = np.concatenate([np.zeros(ndelay), data]).astype(np.complex64)
    padding_data = np.conjugate(padding_data)
    data_bigger = np.abs(data_padding)**2
    data_bigger = np.max([np.abs(data_padding)**2, np.abs(padding_data)**2], axis=0)

    a_list = moving_avg(data_padding * padding_data, window_size=nwindow)
    p_list = moving_avg(data_bigger, window_size=nwindow)

    a_list = np.abs(a_list)
    # print(np.divide(a_list, p_list, out=np.zeros_like(a_list), where=p_list != 0, casting='unsafe'))
    return np.divide(a_list, p_list, out=np.zeros_like(a_list), where=p_list != 0, casting='unsafe')


if __name__=='__main__':
    # test
    data1 = np.array([6,7,8,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,6,7,8], dtype=np.complex)
    data2 = np.array([6,7,8,1,2,3,1.1,2,2.9,1.2,1.8,3,1,2,3.3,6,7,8], dtype=np.complex)
    # print(moving_avg(data1, window_size=6))
    print(acf_norm(data1, ndelay=3, nwindow=6))
    print(acf_norm(data2, ndelay=3, nwindow=6))