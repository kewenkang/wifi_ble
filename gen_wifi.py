# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def gen_t_wifi_symble(f_seq, interval=5e-8, subcarrier_wb=312.5e3):
    assert interval <= 5e-8, 'interval should be less than 1/20M.'
    t = np.arange(0, 4e-6, interval)
    n = len(t)
    t_seq = np.zeros_like(t)
    for i, f in enumerate(f_seq):
        i = f.real
        q = f.imag
        w = 2 * np.pi * subcarrier_wb * i
        subcarrier = i * np.cos(w * t) + q * np.sin(w * t)
        t_seq = t_seq + subcarrier

    t_seq[:int(n/5)] = t_seq[-int(n/5):]    # cycle prefix
    return t, t_seq

def main():
    from util import LONG_F, dofft
    long_t = gen_t_wifi_symble(LONG_F)
    long_t1 = dofft(LONG_F, reverse=True)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(LONG_F.imag, 'b-')
    ax1.plot(LONG_F.real, 'r-')
    ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f1.imag, 'b-')
    ax4.plot(long_f1.real, 'r-')
    plt.show()

def main_s():
    from util import SHORT_F_in_ble1, dofft
    # SHORT_F_in_ble1 = 1
    t, long_t = gen_t_wifi_symble(SHORT_F_in_ble1)
    long_t1 = dofft(SHORT_F_in_ble1, reverse=True)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(SHORT_F_in_ble1.imag, 'b-')
    ax1.plot(SHORT_F_in_ble1.real, 'r-')
    ax2.plot(t, long_t.imag, 'b-')
    ax2.plot(t, long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f1.imag, 'b-')
    ax4.plot(long_f1.real, 'r-')
    plt.show()

def main_l():
    from util import LONG_F_in_ble1, dofft
    # LONG_F_in_ble1 = 1
    t, long_t = gen_t_wifi_symble(LONG_F_in_ble1)
    long_t1 = dofft(LONG_F_in_ble1, reverse=True)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(LONG_F_in_ble1.imag, 'b-')
    ax1.plot(LONG_F_in_ble1.real, 'r-')
    ax2.plot(t, long_t.imag, 'b-')
    ax2.plot(t, long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f1.imag, 'b-')
    ax4.plot(long_f1.real, 'r-')
    plt.show()


def test():
    from util import LONG_F, dofft
    long_t = gen_t_wifi_symble(LONG_F)
    long_t1 = dofft(LONG_F, reverse=True)
    long_f1 = dofft(long_t1, reverse=False)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(LONG_F.imag, 'b-')
    ax1.plot(LONG_F.real, 'r-')
    ax2.plot(long_t.imag, 'b-')
    ax2.plot(long_t.real, 'r-')
    ax3.plot(long_t1.imag, 'b-')
    ax3.plot(long_t1.real, 'r-')
    ax4.plot(long_f1.imag, 'b-')
    ax4.plot(long_f1.real, 'r-')
    plt.show()

if __name__ == '__main__':
    main_l()



