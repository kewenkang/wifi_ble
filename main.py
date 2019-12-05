from __future__ import print_function

import numpy as np
from scipy import signal
from util import dofft

try:
    from pylab import *
except ImportError:
    print("Please install Python Matplotlib (http://matplotlib.sourceforge.net/) and \
           Python TkInter https://wiki.python.org/moin/TkInter to run this script")
    raise SystemExit(1)

def filter_plot():
    hfile = open('./iq_20M.dat', 'rb')
    sample_rate = 20e6
    low_freq = 1e6
    high_freq = 2e6
    high_freq2 = 3e6
    block_size = int(sample_rate*20e-6)
    start = 1000
    datatype = np.complex64
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

    # 带通滤波器
    sos1 = signal.butter(10, [2*low_freq/sample_rate, 2*high_freq/sample_rate], 'bandpass', output='sos')
    sos2 = signal.butter(10, [2*high_freq/sample_rate, 2*high_freq2/sample_rate], 'bandpass', output='sos')
    bandpass_iq = signal.sosfilt(sos1, iq)
    bandpass_reals = np.array([r.real for r in bandpass_iq])
    bandpass_imags = np.array([i.imag for i in bandpass_iq])
    bandpass_iq2 = signal.sosfilt(sos2, iq)
    bandpass_reals2 = np.array([r.real for r in bandpass_iq2])
    bandpass_imags2 = np.array([i.imag for i in bandpass_iq2])

    t = np.linspace(0, 1, int(sample_rate*0.01), False)  # 1 second

    # plot signals
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    plot_iq = ax1.plot(times, reals, 'bo-', times, imags, 'ro-')
    plot_iq1 = ax2.plot(times, bandpass_reals, 'bo-', times, bandpass_imags, 'ro-')
    plot_iq2 = ax3.plot(times, bandpass_reals2, 'bo-', times, bandpass_imags2, 'ro-')

    plt.show()


if __name__=='__main__':
    filter_plot()