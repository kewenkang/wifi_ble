# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import signal
from util import *
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import os
import sys
import time
import logging

# construct fileHandler(and consoleHandler) to logger
logFormatter = logging.Formatter('%(message)s')  # ('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
# fileHandler = logging.FileHandler('./log/log_wifi_%s.txt' % (time.strftime('%Y-%m-%d_%H_%M_%S')), mode='w')
# fileHandler.setFormatter(logFormatter)
# logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.info(time.asctime(time.localtime(time.time())))

np.set_printoptions(linewidth=200)

def filter_plot():
    hfile = open('./data/iq_20M_g.dat', 'rb')
    sample_rate = 20e6
    sample_time = 0.01
    low_freq = 1e6
    high_freq = 2e6
    high_freq2 = 3e6
    block_size = 500000 #int(sample_rate*sample_time)
    min_platue = 2
    start = 6542700 # 0.6e6
    datatype = np.complex64
    threshold = 0.89      # good: 0.1575
    frame_start_indices = [6,132,238]

    global iq

    start_idx = 1000
    window_size = 1000
    wifi_start_idx_list = []
    while True:
        # read iq data from file
        try:
            abs_start = start_idx - window_size
            hfile.seek(datatype().nbytes * abs_start, 0)
            iq = np.fromfile(hfile, dtype=datatype, count=block_size)
            logger.info('start index: %s' % abs_start)
            logger.info('length of iq: %s' % len(iq))

            start_idx += block_size
        except MemoryError:
            logger.info('End of File')
            break
        else:
            # iq signal charactors
            # logger.info('max signal of real part: %.4f' % np.max(reals))


            # autocorrelation
            try:
                ac = acf_norm(iq)
                logger.info('max autocorrelation: %.4f, min: %.4f' % (np.max(ac), np.min(ac)))
                if np.max(ac) >= threshold:

                    # logger.info(ac)
                    # ac_indices = np.arange(len(ac))
                    # ac_indices[ac<threshold] = 0
                    # pd_ac = pd.Series(ac)
                    # logger.info(pd_ac.describe())
                    ac_det_indices = np.arange(len(ac))[ac>threshold]
                    ac_det_start_len = convert_seq2start_len(ac_det_indices)
                    ac[ac < threshold] = 0
                    # logger.info('iq autocorrelation [5]: {}, [6]: {}'.format(ac[5], ac[6]))
                    # logger.info(ac_det_start_len.describe())

                    for idx, row in ac_det_start_len.iterrows():
                        cons_len, rel_idx = row['length'], row['start_index']
                        logger.info('relative index: %s, consecutive length: %s' % (rel_idx, cons_len))
                        if cons_len > min_platue:
                            logger.info('possible wifi signal found at %s' % (abs_start + rel_idx))
                            wifi_start_idx_list.append(abs_start + rel_idx)
            except AssertionError, msg:
                print(msg)
                break
            logger.info('-' * 20)
    logger.info('possible wifi signal length: %s, starts at %s' % (len(wifi_start_idx_list), wifi_start_idx_list))
            # logger.info(ac)







def get_data(offset=192, length=160):
    hfile = open('./data/iq_20M_g.dat', 'rb')
    sample_rate = 20e6
    sample_time = 0.01
    low_freq = 1e6
    high_freq = 2e6
    high_freq2 = 3e6
    block_size = 700  # int(sample_rate*sample_time)
    start = 6542700  # 0.6e6
    datatype = np.complex64
    hfile.seek(datatype().nbytes * start, 1)
    iq = np.fromfile(hfile, dtype=datatype, count=block_size)

    start_index= 35
    iq_slices = iq[start_index + offset: start_index + length + 1 + offset]

    return iq_slices

def plot_data():
    iq = get_data()
    # logger.info(iq)
    iq_fft = dofft(iq)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.abs(iq_fft), 'b-')
    ax2.plot(np.angle(iq_fft), 'b-')
    plt.show()

def test_fir():
    n = 1
    scale = 1/n
    b, a = LONG, [1,1]
    iq = get_data()
    y = signal.filtfilt(b, a, iq, padtype='constant', padlen=4, method='pad')
    plt.plot(np.abs(y), 'b-')
    plt.show()
    i = np.abs(y).argsort()[::-1][:3]
    logger.info(i)



if __name__=='__main__':
    # filter_plot()
    # plot_data()
    test_fir()