from __future__ import print_function

import numpy as np
from numpy.fft import fftpack

def dofft(self, iq):
    N = len(iq)
    iq_fft = np.fft.fftshift(fftpack.fft(iq))  # fft and shift axis
    iq_fft = 20 * np.log10(abs((iq_fft + 1e-15) / N))  # convert to decibels, adjust power
    # adding 1e-15 (-300 dB) to protect against value errors if an item in iq_fft is 0
    return iq_fft