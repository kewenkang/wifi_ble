# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

INTERVAL = 1e-8
NUMBER_PER_MU = int((1e-6)/INTERVAL)

def gen_t_wifi_symble(f_seq, interval=INTERVAL, subcarrier_wb=312.5e3):
    assert interval <= 5e-8, 'interval should be less than 1/20M.'
    t = np.arange(0, 3.2e-6, interval)
    n = len(t)
    t_seq = np.zeros_like(t)
    t_seq_iq = np.zeros_like(t)
    for i, f in enumerate(f_seq):
        ii = f.real
        qq = f.imag
        w = 2 * np.pi * subcarrier_wb * (i + 1)
        subcarrier_ii = ii * np.cos(w * t)
        subcarrier_qq = qq * np.sin(w * t)
        part_seq = 1j * subcarrier_qq
        part_seq = part_seq + subcarrier_ii
        t_seq_iq = t_seq_iq + part_seq

        part_seq = subcarrier_ii + subcarrier_qq
        t_seq = t_seq + part_seq

    # t_seq = np.concatenate((t_seq[-int(n/4):], t_seq))    # cycle prefix
    return t_seq, t_seq_iq

def gen_fsk(bits, fs=1e6, h=0.5, num_per_mu=NUMBER_PER_MU):
    fd = fs * h / 2
    fc = 11e6
    # w1 = 2 * np.pi * (fs + fd)
    # w2 = 2 * np.pi * (fs - fd)
    bb = []
    t = np.arange(0, 1e-6, INTERVAL)
    t_seq = np.array([])
    for b in bits:
        w = 2 * np.pi * (fc + b * fd)
        phi = 0 #np.pi / 4
        sig = np.cos(w * t + phi)
        t_seq = np.concatenate((t_seq, sig))
        bb += [b] * int(num_per_mu)
    # print(bb)
    # bb = ndimage.gaussian_filter1d(np.array(bb), 1)
    # print(bb)
    return bb, t_seq

def main_l():
    from util import LONG_F, dofft
    long_t, long_t_iq = gen_t_wifi_symble(LONG_F)
    long_t1 = dofft(LONG_F, reverse=True)
    long_f = dofft(long_t, reverse=False)
    long_f_iq = dofft(long_t_iq, reverse=False)
    fig, ((ax1, ax3), (ax2, ax4), (ax22, ax22f)) = plt.subplots(3, 2)
    ax1.plot(LONG_F.imag, 'b-')
    ax1.plot(LONG_F.real, 'r-')
    ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax22.plot(long_t_iq.imag, 'b-')
    ax22.plot(long_t_iq.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f.imag, 'b-')
    ax4.plot(long_f.real, 'r-')
    ax22f.plot(long_f_iq.imag, 'b-')
    ax22f.plot(long_f_iq.real, 'r-')
    plt.show()

def main_s():
    from util import SHORT_F, dofft
    long_t, long_t_iq = gen_t_wifi_symble(SHORT_F)
    long_t1 = dofft(SHORT_F, reverse=True)
    long_f = dofft(long_t, reverse=False)
    long_f_iq = dofft(long_t_iq, reverse=False)
    fig, ((ax1, ax3), (ax2, ax4), (ax22, ax22f)) = plt.subplots(3, 2)
    ax1.plot(SHORT_F.imag, 'b-')
    ax1.plot(SHORT_F.real, 'r-')
    ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax22.plot(long_t_iq.imag, 'b-')
    ax22.plot(long_t_iq.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f.imag, 'b-')
    ax4.plot(long_f.real, 'r-')
    ax22f.plot(long_f_iq.imag, 'b-')
    ax22f.plot(long_f_iq.real, 'r-')
    plt.show()

def main_s_part():
    from util import SHORT_F_in_ble1, dofft
    # SHORT_F_in_ble1 = 1
    long_t = gen_t_wifi_symble(SHORT_F_in_ble1)
    long_t1 = dofft(SHORT_F_in_ble1, reverse=True)
    long_f = dofft(long_t, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(SHORT_F_in_ble1.imag, 'b-')
    ax1.plot(SHORT_F_in_ble1.real, 'r-')
    ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f.imag, 'b-')
    ax4.plot(long_f.real, 'r-')
    plt.show()

def main_l_part():
    from util import LONG_F_in_ble1, dofft
    # LONG_F_in_ble1 = 1
    long_t = gen_t_wifi_symble(LONG_F_in_ble1)
    long_t1 = dofft(LONG_F_in_ble1, reverse=True)
    long_f = dofft(long_t, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(LONG_F_in_ble1.imag, 'b-')
    ax1.plot(LONG_F_in_ble1.real, 'r-')
    ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f.imag, 'b-')
    ax4.plot(long_f.real, 'r-')
    plt.show()

def main_l_part_down_sampling():
    from util import LONG_F_in_ble1, dofft
    # long_t = gen_t_wifi_symble(LONG_F_in_ble1)
    long_t1 = dofft(LONG_F_in_ble1, reverse=True)
    long_t1 = np.concatenate((long_t1[-32:], long_t1, long_t1))
    long_t1_down_sampling = long_t1[::10]

    long_f = dofft(long_t1_down_sampling, reverse=False)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax22, ax3, ax4) = plt.subplots(5, 1)
    ax1.plot(LONG_F_in_ble1.imag, 'b-')
    ax1.plot(LONG_F_in_ble1.real, 'r-')
    ax2.plot(long_t1.imag, 'b-')
    ax2.plot(long_t1.real, 'r-')
    ax22.plot(long_f1.imag, 'b-')
    ax22.plot(long_f1.real, 'r-')
    ax3.plot(long_t1_down_sampling.imag, 'b-')
    ax3.plot(long_t1_down_sampling.real, 'r-')
    ax4.plot(long_f.imag, 'b-')
    ax4.plot(long_f.real, 'r-')
    plt.savefig('./pictures/down_sampling_l.png')
    plt.show()

def main_s_part_down_sampling():
    from util import SHORT_F_in_ble1, dofft
    # long_t = gen_t_wifi_symble(LONG_F_in_ble1)
    f = SHORT_F_in_ble1
    long_t1 = dofft(f, reverse=True)
    long_t1 = np.concatenate((long_t1[-32:], long_t1, long_t1))
    long_t1_down_sampling = long_t1[::10]

    long_f = dofft(long_t1_down_sampling, reverse=False)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax22, ax3, ax4) = plt.subplots(5, 1)
    ax1.plot(f.imag, 'b-')
    ax1.plot(f.real, 'r-')
    ax2.plot(long_t1.imag, 'b-')
    ax2.plot(long_t1.real, 'r-')
    ax22.plot(long_f1.imag, 'b-')
    ax22.plot(long_f1.real, 'r-')
    ax3.plot(long_t1_down_sampling.imag, 'b-')
    ax3.plot(long_t1_down_sampling.real, 'r-')
    ax4.plot(long_f.imag, 'b-')
    ax4.plot(long_f.real, 'r-')
    plt.savefig('./pictures/down_sampling_s.png')
    plt.show()


def test_gen_wifi():
    from util import LONG_F, dofft
    f = np.array([complex(5, 3), complex(-7,-3), complex(-1,3), complex(-1, 1)])
    long_t = gen_t_wifi_symble(f)
    long_t1 = dofft(f, reverse=True)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(f.imag, 'b-')
    ax1.plot(f.real, 'r-')
    # ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f1.imag, 'b-')
    ax4.plot(long_f1.real, 'r-')
    plt.show()

def test_fsk():
    from util import dofft
    bits = np.array([1, -1, 1, -1])
    bb, long_t = gen_fsk(bits)
    long_f = dofft(long_t, reverse=False)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(bb, 'b-')
    ax2.plot(long_t, 'r-')
    ax3.plot(long_f.imag, 'b-')
    ax3.plot(long_f.real, 'r-')
    plt.savefig('./pictures/2fsk.png')
    plt.show()

if __name__ == '__main__':
    # main_l()
    # main_s_part_down_sampling()
    # main_l_part_down_sampling()
    test_fsk()



