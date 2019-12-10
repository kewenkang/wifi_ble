# -*- coding: utf8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import signal
from util import *
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

def filter_plot():
    hfile = open('./data/iq_20M_g.dat', 'rb')
    sample_rate = 20e6
    sample_time = 0.01
    low_freq = 1e6
    high_freq = 2e6
    high_freq2 = 3e6
    block_size = 700 #int(sample_rate*sample_time)
    start = 6542700 # 0.6e6
    datatype = np.complex64
    threshold = 0.89      # good: 0.1575
    frame_start_indices = [6,132,238]

    global iq,reals,imags,times

    # read iq data from file
    try:
        hfile.seek(datatype().nbytes*start, 1)
        iq = np.fromfile(hfile, dtype=datatype, count=block_size)
    except MemoryError:
        print('End of File')
    else:
        reals = np.array([r.real for r in iq])
        imags = np.array([i.imag for i in iq])
        times = np.array([i*(1/sample_rate) for i in range(len(reals))])

    # iq signal charactors
    print(np.max(reals), np.argmax(reals))

    # 带通滤波器
    sos1 = signal.butter(10, [2*low_freq/sample_rate, 2*high_freq/sample_rate], 'bandpass', output='sos')
    sos2 = signal.butter(10, [2*high_freq/sample_rate, 2*high_freq2/sample_rate], 'bandpass', output='sos')
    bandpass_iq = signal.sosfilt(sos1, iq)
    bandpass_reals = np.array([r.real for r in bandpass_iq])
    bandpass_imags = np.array([i.imag for i in bandpass_iq])
    bandpass_iq2 = signal.sosfilt(sos2, iq)
    bandpass_reals2 = np.array([r.real for r in bandpass_iq2])
    bandpass_imags2 = np.array([i.imag for i in bandpass_iq2])
    print('iq length:{}, iq1 length:{}, iq2 length:{}'.format(len(iq), len(bandpass_iq), len(bandpass_iq2)))


    # autocorrelation
    ac = acf_norm(iq)
    # print(ac)
    ac_indices = np.arange(len(ac))
    ac_indices[ac<threshold] = 0
    pd_ac = pd.Series(ac)
    print(pd_ac.describe())
    ac_det_indices = np.arange(len(ac))[ac>threshold]
    ac_det_start_len = convert_seq2start_len(ac_det_indices)
    ac[ac < threshold] = 0
    print('iq autocorrelation [5]: {}, [6]: {}'.format(ac[5], ac[6]))
    print(ac_det_start_len.describe())
    print(ac_det_start_len)
    print(ac)

    # fft frequency domain analysis
    start_index, length, offset = 35, 64, 192+64
    iq_slices = iq[start_index+offset: start_index+length+1+offset]
    print(iq_slices)
    iq_fft = dofft(iq_slices)
    tstep = 1.0 / sample_rate
    # self.time = numpy.array([tstep*(self.position + i) for i in range(len(self.iq))])
    time = np.array([tstep * (i) for i in range(len(iq_slices))])
    freqs = calc_freq(time, sample_rate)

    # plot signals
    fig, (ax1_t, ax1_s, ax1_f) = plt.subplots(3, 1)
    plot_iq = ax1_t.plot(times, reals, 'b-', times, imags, 'r-')
    # plot_iq1 = ax2.plot(times, bandpass_reals, 'bo-', times, bandpass_imags, 'ro-')
    # plot_iq2 = ax3.plot(times, bandpass_reals2, 'bo-', times, bandpass_imags2, 'ro-')
    # plot fft
    ax1_s.plot(time, iq_slices.real, 'b-')
    ax1_f.plot(freqs, np.abs(iq_fft), 'b-')
    # ax1_f.set_ylim((0,1.1))
    plt.show()



    # import statsmodels.tsa.api as smt



def get_data():
    hfile = open('./data/iq_20M_g.dat', 'rb')
    datatype = np.complex64
    block_size = 100000
    iq = np.fromfile(hfile, dtype=datatype, count=block_size)
    return iq

def pretreat():
    iq = get_data()

    #
    ac = acf_norm(iq)
    ac_indices = np.arange(len(ac))
    ac_indices[ac < threshold] = 0
    pd_ac = pd.Series(ac)
    print(pd_ac.describe())
    ac_det_indices = np.arange(len(ac))[ac > threshold]
    ac_det_start_len = convert_seq2start_len(ac_det_indices)
    ac[ac < threshold] = 0
    print('iq autocorrelation [5]: {}, [6]: {}'.format(ac[5], ac[6]))
    print(ac_det_start_len.describe())
    print(ac_det_start_len)

if __name__=='__main__':
    filter_plot()