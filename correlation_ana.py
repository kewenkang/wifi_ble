# -*- coding: utf8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import signal
from util import *
import sys
import time
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

logFormatter = logging.Formatter('%(message)s')
logger = logging.getLogger('FIND_WIFI')
logger.setLevel(logging.INFO)
log_filename = './log/log_cor.txt'
# log_filename = './log/log_cor_{}.txt'.format(time.strftime('%Y-%m-%d_%H_%M_%S'))
fileHandler = logging.FileHandler(log_filename, mode='w', encoding='utf8')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

def correlation_ana():
    hfile = open('./data/iq_20M_g.dat', 'rb')
    sample_rate = 20e6
    sample_time = 0.01
    low_freq = 2e6
    high_freq = 3e6
    high_freq2 = 4e6
    high_freq3 = 5e6
    # slice1 = [26, 27, 28, 30, 35, 36, 37]
    # slice2 = [23, 24, 25, 30, 38, 39, 40]
    slice1 = [23, 24, 25, 38, 39, 40]
    slice2 = [19, 20, 21, 41, 43, 44]
    slice3 = [16, 17, 18, 45, 46, 47]

    block_size = 1000 #int(sample_rate*sample_time)
    start = 17697015 # 0.6e6
    datatype = np.complex64
    threshold = 0.89      # good: 0.1575

    global iq,reals,imags,times
    win_size = 300
    frame_start_indices = [6542735, 15343724, 17672171, 17673993, 17674551, 17694631, 17696454, 17697015, 17730220, 17737318, 19202891, 19203766, 19204326, 22753822, 22928923, 25608012, 25608573, 34472883, 34473444, 37525336, 37583039, 37583599, 37674779, 37694316, 37696139, 37696223, 37696708, 37715403, 37760957, 37825994, 37829304, 37831677, 37839288, 37844740, 37850192, 37966577, 37967136, 38036997, 38043220, 38045044, 38045604, 38068410, 38070234, 38070794, 38096119, 38097941, 38098499, 38259436, 38259996, 38535625, 38537448, 38538008, 38564478, 38754634, 38953652, 39573648, 39574206, 39587213, 40518050, 41424594, 41427564, 41452744, 41453301, 41463956, 41464513, 41479051, 41479608, 41485084, 41485642, 41863616, 41864176, 42227548, 42228423, 42228983, 44365157, 44369785, 44376559, 44383668, 44388417, 44393067]

    wifi_csi1_mags = []
    wifi_csi2_mags = []
    wifi_csi3_mags = []
    ble_channel1_mags = []
    ble_channel2_mags = []
    ble_channel3_mags = []

    for fs_idx in frame_start_indices:
        '''read iq data from file'''
        try:
            hfile.seek(datatype().nbytes * fs_idx, 0)
            iq = np.fromfile(hfile, dtype=datatype, count=block_size)
        except MemoryError:
            logger.warning('End of File')
        else:
            reals = np.array([r.real for r in iq])
            imags = np.array([i.imag for i in iq])
            times = np.array([i*(1/sample_rate) for i in range(len(reals))])

        # iq signal charactors
        logger.info('max real: %s, min real: %s, mean real: %s' % (np.max(np.abs(reals)), np.min(np.abs(reals)), np.mean(np.abs(reals))))

        '''带通滤波器'''
        ble_offset, ble_len = 500, 64
        sos1 = signal.butter(10, [2*low_freq/sample_rate, 2*high_freq/sample_rate], 'bandpass', output='sos')
        sos2 = signal.butter(10, [2*high_freq/sample_rate, 2*high_freq2/sample_rate], 'bandpass', output='sos')
        sos3 = signal.butter(10, [2*high_freq2/sample_rate, 2*high_freq3/sample_rate], 'bandpass', output='sos')

        bandpass_iq = signal.sosfilt(sos1, iq)
        bandpass_reals = np.array([r.real for r in bandpass_iq])
        bandpass_imags = np.array([i.imag for i in bandpass_iq])
        ble_c1_rssi = np.sum(np.abs(bandpass_iq[ble_offset: ble_offset + ble_len]))
        ble_channel1_mags.append(ble_c1_rssi)
        # print(ble_c1_rssi)

        bandpass_iq2 = signal.sosfilt(sos2, iq)
        ble_c2_rssi = np.sum(np.abs(bandpass_iq2[ble_offset: ble_offset + ble_len]))
        ble_channel2_mags.append(ble_c2_rssi)

        bandpass_iq3 = signal.sosfilt(sos3, iq)
        ble_c3_rssi = np.sum(np.abs(bandpass_iq3[ble_offset: ble_offset + ble_len]))
        ble_channel3_mags.append(ble_c3_rssi)

        # print('iq length:{}, iq1 length:{}, iq2 length:{}'.format(len(iq), len(bandpass_iq), len(bandpass_iq2)))




        '''FFT frequency domain analysis'''
        start_index, length, offset = 0, 64, 192
        iq_slices = iq[start_index+offset: start_index+length+offset]
        # print(iq_slices)
        iq_fft = dofft(iq_slices)
        csi1 = np.sum(np.abs(iq_fft[slice1]))
        csi2 = np.sum(np.abs(iq_fft[slice2]))
        csi3 = np.sum(np.abs(iq_fft[slice3]))
        # print(csi1)
        wifi_csi1_mags.append(csi1)
        wifi_csi2_mags.append(csi2)
        wifi_csi3_mags.append(csi3)

        # self.time = numpy.array([tstep*(self.position + i) for i in range(len(self.iq))])

        '''plot signals'''
        # tstep = 1.0 / sample_rate
        # time = np.array([tstep * (i) for i in range(len(iq_slices))])
        # freqs = calc_freq(time, sample_rate)
        # fig, (ax1_t, ax1_s, ax1_f) = plt.subplots(3, 1)
        # ax1_t.plot(times, reals, 'b-', times, imags, 'r-')
        # # plot fft
        # ax1_s.plot(time, iq_slices.real, 'b-')
        # ax1_f.plot(freqs, np.abs(iq_fft), 'b-')

        # fig, (ax1_t, ax2, ax3) = plt.subplots(3, 1)
        # ax1_t.plot(times, reals, 'b-', times, imags, 'r-')
        # plot_iq1 = ax2.plot(times, bandpass_reals, 'bo-', times, bandpass_imags, 'ro-')
        # plot_iq2 = ax3.plot(times, bandpass_reals2, 'bo-', times, bandpass_imags2, 'ro-')
        #
        # # ax1_f.set_ylim((0,1.1))
        # plt.show()

    # print(wifi_csi1_mags)
    # print(ble_channel1_mags)

    print(max(ble_channel1_mags), ble_channel1_mags.index(min(ble_channel1_mags)))
    print(max(ble_channel2_mags), ble_channel2_mags.index(min(ble_channel2_mags)))
    print(max(ble_channel3_mags), ble_channel3_mags.index(min(ble_channel3_mags)))
    print(max(wifi_csi1_mags), wifi_csi1_mags.index(min(wifi_csi1_mags)))
    print(max(wifi_csi2_mags), wifi_csi2_mags.index(min(wifi_csi2_mags)))
    print(max(wifi_csi3_mags), wifi_csi3_mags.index(min(wifi_csi3_mags)))


    '''correlation calculation'''
    csi_rssi_1 = np.array([wifi_csi1_mags, ble_channel1_mags])
    csi_rssi_2 = np.array([wifi_csi2_mags, ble_channel2_mags])
    csi_rssi_3 = np.array([wifi_csi3_mags, ble_channel3_mags])

    for cut_len in range(2, 80):
        print(cut_len)
        csi_list = [wifi_csi1_mags[:cut_len], wifi_csi2_mags[:cut_len], wifi_csi3_mags[:cut_len]]
        rssi_list = [ble_channel1_mags[:cut_len], ble_channel2_mags[:cut_len], ble_channel3_mags[:cut_len]]

        '''order relationship'''
        for i, (csi, rssi) in enumerate(zip(csi_list, rssi_list)):
            # print(csi, rssi)
            # print(get_order(csi), get_order(rssi))
            # print(zero_one_norm(csi), zero_one_norm(rssi))
            # print(zscore_norm(csi), zscore_norm(rssi))
            print('raw data correlation: %s' % np.corrcoef(np.array([csi, rssi]))[0][1])
            print('order correlation:    %s' % np.corrcoef(np.array([get_order(csi), get_order(rssi)]))[0][1])
            print('zero one correlation: %s' % np.corrcoef(np.array([zero_one_norm(csi), zero_one_norm(rssi)]))[0][1])
            print('zscore correlation:   %s' % np.corrcoef(np.array([zscore_norm(csi), zscore_norm(rssi)]))[0][1])

    ''' relationship between correlation and seq_len '''
    length_list = list(range(5, 80, 1))
    cor_len_list = []
    for l in length_list:
        cor = np.corrcoef(np.array([wifi_csi1_mags[:l], ble_channel1_mags[:l]]))[0][1]
        cor_len_list.append(cor)
        # logger.info('correlation (length: %s): %s' % (l, cor))

    plt.plot(length_list, cor_len_list)
    plt.xlabel('CSI&RSSI sequence length')
    plt.ylabel('Correlation coefficient')
    plt.savefig('./pictures/cor_length.png')
    # plt.show()

    '''plot LSM'''
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    # m, c = LSM(ble_channel1_mags, wifi_csi1_mags)
    # x = np.arange(0, 3, 0.1)
    # y = m * x + c
    # ax1.scatter(ble_channel1_mags, wifi_csi1_mags)
    # ax1.plot(x, y, 'r')
    # ax1.set_xlabel('RSSI 1 (r=0.78)')
    # ax1.set_ylabel('CSI 1')
    #
    # m, c = LSM(ble_channel2_mags, wifi_csi2_mags)
    # x = np.arange(0, 3, 0.1)
    # y = m * x + c
    # ax2.scatter(ble_channel2_mags, wifi_csi2_mags)
    # ax2.plot(x, y, 'r')
    # ax2.set_xlabel('RSSI 2 (r=0.71)')
    # ax2.set_ylabel('CSI 2')
    #
    # m, c = LSM(ble_channel3_mags, wifi_csi3_mags)
    # x = np.arange(0, 3, 0.1)
    # y = m * x + c
    # ax3.scatter(ble_channel3_mags, wifi_csi3_mags)
    # ax3.plot(x, y, 'r')
    # ax3.set_xlabel('RSSI 3 (r=0.72)')
    # ax3.set_ylabel('CSI 3')
    # plt.tight_layout()
    # plt.savefig('./pictures/LSM.png', bbox_inches='tight')
    # plt.show()

    '''csi contrasts rssi'''
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # min_ylim, max_ylim = 0, 7.5
    # ax1.set_ylim(min_ylim, max_ylim)
    # ax2.set_ylim(min_ylim, max_ylim)
    # ax3.set_ylim(min_ylim, max_ylim)
    #
    # indices = range(len(frame_start_indices))
    # ax1.plot(indices, wifi_csi1_mags, 'r-', label='CSI')
    # ax1.plot(indices, ble_channel1_mags, 'b-', label='RSSI')
    # ax1.set_title('channel1')
    # ax1.legend(loc='upper right')
    #
    # ax2.plot(indices, wifi_csi2_mags, 'r-', label='CSI')
    # ax2.plot(indices, ble_channel2_mags, 'b-', label='RSSI')
    # ax2.set_title('channel2')
    # ax2.legend(loc='upper right')
    #
    # ax3.plot(indices, wifi_csi3_mags, 'r-', label='CSI')
    # ax3.plot(indices, ble_channel3_mags, 'b-', label='RSSI')
    # ax3.set_title('channel3')
    # ax3.legend(loc='upper right')
    # plt.tight_layout()
    # plt.savefig('./pictures/csi&rssi.png', bbox_inches='tight')
    # plt.show()

    '''csi contrasts csi, rssi contrasts rssi'''
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax1.set_ylim(min_ylim, max_ylim)
    # indices = range(len(frame_start_indices))
    # ax1.plot(indices, wifi_csi1_mags, 'r-', label='CSI1')
    # ax1.plot(indices, wifi_csi2_mags, 'g-', label='CSI2')
    # ax1.plot(indices, wifi_csi3_mags, 'b-', label='CSI3')
    # ax1.set_title('CSI')
    # ax1.legend(loc='upper right')
    # ax2.plot(indices, ble_channel1_mags, 'r-', label='RSSI1')
    # ax2.plot(indices, ble_channel2_mags, 'g-', label='RSSI2')
    # ax2.plot(indices, ble_channel3_mags, 'b-', label='RSSI3')
    # ax2.set_title('RSSI')
    # # ax2.set_xlabel('index')
    # ax2.legend(loc='upper right')
    # plt.tight_layout()
    # plt.savefig('./pictures/csis_rssis.png', bbox_inches='tight')
    # plt.show()







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
    # print(iq)
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
    print(i)



if __name__=='__main__':
    correlation_ana()
    # plot_data()
    # test_fir()