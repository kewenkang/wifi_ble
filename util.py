# from __future__ import print_function

import numpy as np
import pandas as pd
from numpy.fft import fftpack
# import matplotlib.pyplot as plt


LONG = np.array([
complex(-0.0455, -1.0679), complex( 0.3528, -0.9865), complex( 0.8594,  0.7348), complex( 0.1874,  0.2475),
complex( 0.5309, -0.7784), complex(-1.0218, -0.4897), complex(-0.3401, -0.9423), complex( 0.8657, -0.2298),
complex( 0.4734,  0.0362), complex( 0.0088, -1.0207), complex(-1.2142, -0.4205), complex( 0.2172, -0.5195),
complex( 0.5207, -0.1326), complex(-0.1995,  1.4259), complex( 1.0583, -0.0363), complex( 0.5547, -0.5547),
complex( 0.3277,  0.8728), complex(-0.5077,  0.3488), complex(-1.1650,  0.5789), complex( 0.7297,  0.8197),
complex( 0.6173,  0.1253), complex(-0.5353,  0.7214), complex(-0.5011, -0.1935), complex(-0.3110, -1.3392),
complex(-1.0818, -0.1470), complex(-1.1300, -0.1820), complex( 0.6663, -0.6571), complex(-0.0249,  0.4773),
complex(-0.8155,  1.0218), complex( 0.8140,  0.9396), complex( 0.1090,  0.8662), complex(-1.3868, -0.0000),
complex( 0.1090, -0.8662), complex( 0.8140, -0.9396), complex(-0.8155, -1.0218), complex(-0.0249, -0.4773),
complex( 0.6663,  0.6571), complex(-1.1300,  0.1820), complex(-1.0818,  0.1470), complex(-0.3110,  1.3392),
complex(-0.5011,  0.1935), complex(-0.5353, -0.7214), complex( 0.6173, -0.1253), complex( 0.7297, -0.8197),
complex(-1.1650, -0.5789), complex(-0.5077, -0.3488), complex( 0.3277, -0.8728), complex( 0.5547,  0.5547),
complex( 1.0583,  0.0363), complex(-0.1995, -1.4259), complex( 0.5207,  0.1326), complex( 0.2172,  0.5195),
complex(-1.2142,  0.4205), complex( 0.0088,  1.0207), complex( 0.4734, -0.0362), complex( 0.8657,  0.2298),
complex(-0.3401,  0.9423), complex(-1.0218,  0.4897), complex( 0.5309,  0.7784), complex( 0.1874, -0.2475),
complex( 0.8594, -0.7348), complex( 0.3528,  0.9865), complex(-0.0455,  1.0679), complex( 1.3868, -0.0000),
])

LONG_F = np.array([0,0,0,0,0,0,
                   1,  1, -1, -1, 1,  1, -1,  1, -1,  1,  1,  1,  1,  1, 1, -1, -1,  1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
                   1, -1, -1,  1, 1, -1,  1, -1,  1, -1, -1, -1, -1, -1, 1,  1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
                   0,0,0,0,0])

LONG_F_in_ble1 = np.array([0,0,0,0,0,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0, 0, 1, -1, -1,  1, 1,
                    0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,
                    0,0,0,0,0
                    ])

LONG_F_in_ble1_only = np.array([0, 1, -1, -1,  1, 1])


SHORT_F = np.array([0,0,0,0,0,0,0,0,
                    complex(1,1)    ,0,0,0,
                    complex(-1,-1)  ,0,0,0,
                    complex(1,1)    ,0,0,0,
                    complex(-1,-1)  ,0,0,0,
                    complex(-1,-1)  ,0,0,0,
                    complex(1,1)    ,0,0,0,0,0,0,0,
                    complex(-1,-1)  ,0,0,0,
                    complex(-1,-1)  ,0,0,0,
                    complex(1,1)    ,0,0,0,
                    complex(1,1)    ,0,0,0,
                    complex(1,1)    ,0,0,0,
                    complex(1,1)    ,0,0,
                    0,0,0,0,0
                    ])

SHORT_F_in_ble1 = np.array([0,0,0,0,0,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0, 0,0,0,0,
                    complex(-1,-1)  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,0,
                    0  ,0,0,
                    0,0,0,0,0
                    ])

SHORT_F_in_ble1_only = np.array([0,0,0,0,complex(-1,-1)  ,0])

def calc_freq(time, sample_rate):
    N = len(time)
    Fs = 1.0 / (max(time) - min(time))
    Fn = 0.5 * sample_rate
    freq = np.array([-Fn + i * Fs for i in range(N)])
    return freq

def dofft(iq, reverse=False):
    if reverse:
        # iq_ifft = np.fft.ifftshift(fftpack.ifft(iq))
        iq_ifft = fftpack.ifft(iq)
        return iq_ifft
    else:
        # iq_fft = np.fft.fftshift(fftpack.fft(iq))  # fft and shift axis
        iq_fft = fftpack.fft(iq)  # fft and shift axis
        # iq_fft = fftpack.fft(iq)  # fft no shift axis
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
    assert len(data) >= nwindow, 'the length of data should be longger then nwindow'
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

# def plot_long():
#     f_long = dofft(LONG)
#     fig, (ax1, ax2) = plt.subplots(2,1)
#     ax1.plot(np.abs(f_long), 'b-')
#     ax2.plot(np.angle(f_long), 'r-')
#     plt.show()

def LSM(xcord, ycord):
    xcord, ycord = np.array(xcord), np.array(ycord)
    m = ((xcord * ycord).mean() - xcord.mean() * ycord.mean()) / (np.power(xcord, 2).mean() - np.power(xcord.mean(), 2))
    c = ycord.mean() - m * xcord.mean()
    return m, c

def get_order(seq):
    argsort_seq = np.argsort(seq)
    seq_order = np.zeros_like(argsort_seq)
    for ii, i in enumerate(argsort_seq):
        seq_order[i] = ii
    return seq_order

def zero_one_norm(seq):
    seq = np.array(seq, dtype=np.float)
    return (seq - np.min(seq))/(np.max(seq) - np.min(seq))

def avg_norm(seq):
    seq = np.array(seq, dtype=np.float)
    return (seq - np.mean(seq))/(np.max(seq) - np.min(seq))

def zscore_norm(seq):
    seq = np.array(seq, dtype=np.float)
    return (seq - np.mean(seq))/np.std(seq)

if __name__=='__main__':
    # test
    # plot_long()
    data1 = np.array([6,7,8,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,6,7,8], dtype=np.complex)
    data2 = np.array([6,7,8,1,2,3,1.1,2,2.9,1.2,1.8,3,1,2,3.3,6,7,8], dtype=np.complex)
    # print(moving_avg(data1, window_size=6))
    # print(acf_norm(data1, ndelay=3, nwindow=6))
    # print(acf_norm(data2, ndelay=3, nwindow=6))
    print(zero_one_norm([11,9,10,8]))
    print(avg_norm([11,9,10,8]))
    print(zscore_norm([11,9,10,8]))