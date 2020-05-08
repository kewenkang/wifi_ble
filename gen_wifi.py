# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

from gen_ble import *

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



def gfsk_demod_test():
    from util import SHORT_F_in_ble1, LONG_F_in_ble1, dofft
    # long_t = gen_t_wifi_symble(LONG_F_in_ble1)
    f = LONG_F_in_ble1
    long_t1 = dofft(f, reverse=True)
    long_t1 = np.concatenate((long_t1[-32:], long_t1, long_t1))
    long_t1_down_sampling = long_t1[::10]
    # print(long_t1_down_sampling)
    # iq_t = long_t1_down_sampling
    # for i in range(1, len(iq_t)):
    #     s1 = iq_t[i-1]
    #     s2 = iq_t[i]
    #     ang_ = np.angle(s1.conjugate() * s2)
    #     if ang_ >= 0:
    #         print(1)
    #     else:
    #         print(0)
    #     print('delta phi: %s' % ( ang_ / np.pi))
    print(gfsk_demod(long_t1_down_sampling))

def ble_demod():
    from util import LONG_F, dofft
    subcarrier_wb = 0.3125
    subcarrier_f = np.arange(subcarrier_wb/2, 20, subcarrier_wb)
    # print(len(subcarrier_f))
    for i in range(0, 20, 2):
        ble_begin = i
        ble_end = i + 2
        ble_channel_id = ble_end/2
        LONG_in_ble = np.where((subcarrier_f > ble_begin) & (subcarrier_f < ble_end), LONG_F, 0)
        long_t_in_ble = dofft(LONG_in_ble, reverse=True)
        long_t = np.concatenate((long_t_in_ble[-32:], long_t_in_ble, long_t_in_ble))
        for j in range(20):
            long_t_down_sampling = long_t[j::20]
            bits, angs = gfsk_demod(long_t_down_sampling)
            flag_continuous_same = False
            max_diff = 0
            for i, b in enumerate(bits[1:]):
                if bits[i] == b:
                    flag_continuous_same = True
                    # print('not preamble. ble channel id: %d, offset: %d' % (ble_channel_id, j))
                    # break
                else:
                    max_diff += 1
            if not flag_continuous_same:
                print('ble channel id: %d, offset: %d, corresponding bits: %s' % (ble_channel_id, j, bits))
            else:
                if max_diff >=5:
                    # print('max_diff: %d' % max_diff)
                    print(angs)
                    print('ble channel id: %d, offset: %d, corresponding bits: %s' % (ble_channel_id, j, bits))


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

def test_psk():
    I = [1,0,-1,0,-1]
    Q = [0,1,0,-1,0]
    i_q = list(zip(I,Q))
    for i in range(1, len(i_q)):
        s1 = complex(*i_q[i-1])
        s2 = complex(*i_q[i])
        ang_ = np.angle(s1.conjugate() * s2)
        ang1 = np.angle(s1) % (2 * np.pi)
        ang2 = np.angle(s2) % (2 * np.pi)
        # print(ang1, ang2)
        # ang = ang2 - ang1
        ang = (ang2 - ang1)
        print(ang / np.pi)
        print(ang_ / np.pi)

if __name__ == '__main__':
    # main_l()
    # main_s_part_down_sampling()
    # main_l_part_down_sampling()
    # test_fsk()
    # test()
    # gfsk_demod_test()
    ble_demod()
