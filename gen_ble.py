# -*- coding: utf-8 -*-
import numpy as np
import numpy.fft as fft
import scipy.interpolate as itp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from util import zscore_norm, avg_norm

INTERVAL = 1e-8
NUMBER_PER_MU = int((1e-6)/INTERVAL)
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

state_mapping = {
    0: (-1, 0),
    1: (0, -1),
    2: (1, 0),
    3: (0, 1)
}

def bits_to_qpsk_wave(bits, n=80):
    sin_wave = np.sin(np.linspace(0, 2*np.pi, n, endpoint=False))
    res_i = []
    res_q = []
    for i, b in enumerate(bits):
        if i % 2 == 0:
            if b == 0:
                res_i += sin_wave[n/2:].tolist()
            else:
                res_i += sin_wave[:n/2].tolist()
        else:
            if b == 0:
                res_q += sin_wave[n/2:].tolist()
            else:
                res_q += sin_wave[:n/2].tolist()
    res_i = res_i[n/4:] + [0]*(n/4)
    return [complex(i, j) for i, j in zip(res_i, res_q)]

def bits_to_oqpsk(bits):
    bits = np.array(bits)
    bits = np.where(bits==0, -1, 1)
    res_i = []
    res_q = [0]
    for index, b in enumerate(bits):
        if index % 2 == 0:
            res_i.extend([b, 0])
        else:
            res_q.extend([b, 0])
    res_q.pop()
    return res_i, res_q

def gfsk_demod(iq_t):
    bits = []
    angs = []
    if iq_t[0].real > 0:
        bits.append(1)
    else:
        bits.append(0)
    for i in range(1, len(iq_t)):
        s1 = iq_t[i-1]
        s2 = iq_t[i]
        s1 = s1 / abs(s1)
        s2 = s2 / abs(s2)
        ang_ = np.angle(np.arctan(s1.conjugate() * s2))
        # np.arctan()
        # print ang_
        if ang_ >= 0:
            bits.append(1)
        else:
            bits.append(0)
        angs.append(ang_)
    return bits, angs


def main_gen_oqpsk():
    bits = [1,1,0,1,1,0,0,1]
    oqpsk_i, oqpsk_q = bits_to_oqpsk(bits)
    iq_t = []
    for i, q in zip(oqpsk_i, oqpsk_q):
        iq_t.append(complex(i, q))
    print(iq_t)

def find_nearest(points, constellation=[-7, -5, -3, -1, 1, 3, 5, 7]):
    nearests = []
    for p in points:
        nearest = [constellation[0], constellation[0]]
        distance = [np.abs(p[0] - constellation[0]), np.abs(p[1] - constellation[0])]
        for c in constellation[1:]:
            dist0 = np.abs(p[0] - c)
            dist1 = np.abs(p[1] - c)
            if dist0 < distance[0]:
                distance[0] = dist0
                nearest[0] = c
            if dist1 < distance[1]:
                distance[1] = dist1
                nearest[1] = c
        nearests.append(nearest)
    return nearests


def main_demod_oqpsk():
    bits = [1, 0, 1, 0, 1, 0, 1, 0]
    iq_t = [1, -1j, 1, -1j, 1, -1j, 1, -1j]
    constellation = [-7, -5, -3, -1, 1, 3, 5, 7]
    constellation_x, constellation_y = np.meshgrid(constellation, constellation)
    t_index = np.arange(8)
    demod_bits, _ = gfsk_demod(iq_t)
    print demod_bits

    x_smooth = np.linspace(0, 8, 160, endpoint=False)
    # print x_smooth
    iq_smooth = itp.make_interp_spline(t_index, iq_t)(x_smooth)
    # print iq_smooth

    iq_t_gen = bits_to_qpsk_wave(bits)
    iq_t_gen_down_sample = iq_t_gen[::20]

    symble1 = iq_t_gen[16:80]
    symble2 = iq_t_gen[96:]

    f_1 = fft.fftshift(fft.fft(symble1))
    t_1 = fft.ifft(fft.ifftshift(f_1))

    normalized_f_1_real = zscore_norm(np.real(f_1))
    normalized_f_1_iamg = zscore_norm(np.imag(f_1))
    normalized_f_1 = [complex(i, j) for i,j in zip(normalized_f_1_real, normalized_f_1_iamg)]
    normalized_t_1 = fft.ifft(fft.ifftshift(normalized_f_1))

    nearest_f1_points = find_nearest([[i, j] for i,j in zip(normalized_f_1_real, normalized_f_1_iamg)])
    nearest_f1 = [complex(*p) for p in nearest_f1_points]
    nearest_t_1 = fft.ifft(fft.ifftshift(nearest_f1))

    emulated_ble_t = np.concatenate([nearest_t_1[-16:], nearest_t_1, nearest_t_1[-16:], nearest_t_1], axis=0)
    emulated_ble_t_down_sample = emulated_ble_t[::20]
    emulated_demod_bits, _ = gfsk_demod(emulated_ble_t_down_sample)
    print(emulated_demod_bits)


    avg_norm_f_1_real = avg_norm(np.real(f_1))*7
    avg_norm_f_1_iamg = avg_norm(np.imag(f_1))*7
    avg_norm_f_1 = [complex(i, j) for i,j in zip(avg_norm_f_1_real, avg_norm_f_1_iamg)]
    avg_norm_t_1 = fft.ifft(fft.ifftshift(avg_norm_f_1))


    # plt.plot(t_index, iq_t)
    row, col = 4, 3
    plt.subplot(row, col, 1)
    plt.plot(np.arange(len(iq_t_gen)), np.imag(iq_t_gen))
    plt.plot(np.arange(len(iq_t_gen)), np.real(iq_t_gen))
    plt.title('MSK signal')
    plt.subplot(row, col, 2)
    # plt.plot(np.arange(len(iq_t_gen_down_sample)), np.imag(iq_t_gen_down_sample))
    # plt.plot(np.arange(len(iq_t_gen_down_sample)), np.real(iq_t_gen_down_sample))
    # plt.plot(np.arange(len(t_1)), np.imag(t_1))
    # plt.plot(np.arange(len(t_1)), np.real(t_1))
    plt.plot(np.arange(len(symble1)), np.imag(symble1))
    plt.plot(np.arange(len(symble1)), np.real(symble1))
    plt.title('needed wifi symble')
    plt.subplot(row, col, 3)
    plt.plot(np.arange(len(f_1)), np.imag(f_1))
    plt.plot(np.arange(len(f_1)), np.real(f_1))
    plt.title('frequency domain')

    plt.subplot(row, col, 4)
    plt.scatter(normalized_f_1_real, normalized_f_1_iamg, s=4)
    plt.scatter(constellation_x, constellation_y, c='red', s=1)
    plt.grid()
    plt.subplot(row, col, 5)
    plt.plot(np.arange(len(normalized_t_1)), np.imag(normalized_t_1))
    plt.plot(np.arange(len(normalized_t_1)), np.real(normalized_t_1))
    plt.subplot(row, col, 6)
    plt.plot(np.arange(len(normalized_f_1_iamg)), normalized_f_1_iamg)
    plt.plot(np.arange(len(normalized_f_1_real)), normalized_f_1_real)

    # plt.subplot(row, col, 7)
    # plt.scatter(avg_norm_f_1_real, avg_norm_f_1_iamg, s=4)
    # plt.scatter(constellation_x, constellation_y, c='red', s=1)
    # plt.grid()
    # plt.subplot(row, col, 8)
    # plt.plot(np.arange(len(avg_norm_t_1)), np.imag(avg_norm_t_1))
    # plt.plot(np.arange(len(avg_norm_t_1)), np.real(avg_norm_t_1))
    # plt.subplot(row, col, 9)
    # plt.plot(np.arange(len(avg_norm_f_1_iamg)), avg_norm_f_1_iamg)
    # plt.plot(np.arange(len(avg_norm_f_1_real)), avg_norm_f_1_real)

    plt.subplot(row, col, 7)
    plt.scatter(np.real(nearest_f1), np.imag(nearest_f1), s=4)
    plt.scatter(constellation_x, constellation_y, c='red', s=1)
    plt.grid()
    plt.subplot(row, col, 8)
    plt.plot(np.arange(len(nearest_t_1)), np.imag(nearest_t_1))
    plt.plot(np.arange(len(nearest_t_1)), np.real(nearest_t_1))
    plt.subplot(row, col, 9)
    plt.plot(np.arange(len(nearest_f1)), np.imag(nearest_f1))
    plt.plot(np.arange(len(nearest_f1)), np.real(nearest_f1))

    plt.subplot(row, col, 10)
    plt.plot(np.arange(len(emulated_ble_t)), np.imag(emulated_ble_t))
    plt.plot(np.arange(len(emulated_ble_t)), np.real(emulated_ble_t))
    plt.subplot(row, col, 11)
    plt.plot(np.arange(len(emulated_ble_t_down_sample)), np.imag(emulated_ble_t_down_sample))
    plt.plot(np.arange(len(emulated_ble_t_down_sample)), np.real(emulated_ble_t_down_sample))


    plt.tight_layout()
    plt.show()


def test_nearest():
    points = [(2,3), (1,4), (1.5,7)]
    print(find_nearest(points))

if __name__ == '__main__':
    main_demod_oqpsk()
    # test_nearest()





















