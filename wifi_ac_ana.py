# -*- coding: utf8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from scipy import signal
from argparse import ArgumentParser

np.set_printoptions(linewidth=200, threshold='nan')

try:
    from pylab import *
except ImportError:
    print("Please install Python Matplotlib (http://matplotlib.sourceforge.net/) and \
           Python TkInter https://wiki.python.org/moin/TkInter to run this script")
    raise SystemExit(1)

class wifi_ana(object):

    def __init__(self, filename, options):
        self.hfile = open(filename, 'rb')
        self.start = options.start
        self.sample_rate = options.sample_rate
        self.block_length = options.block

        self.datatype = np.complex64
        self.sizeof_data = self.datatype().nbytes

        self.axis_font_size = 16
        self.label_font_size = 18
        self.title_font_size = 20
        self.text_size = 22

        # Setup PLOT
        self.fig = figure(1, figsize=(16, 9), facecolor='w')
        rcParams['xtick.labelsize'] = self.axis_font_size
        rcParams['ytick.labelsize'] = self.axis_font_size

        self.text_file = figtext(0.10, 0.94, ("File: %s" % filename), weight="heavy", size=self.text_size)
        self.text_file_pos = figtext(0.10, 0.88, "File Position: ", weight="heavy", size=self.text_size)
        self.text_block = figtext(0.35, 0.88, ("Block Size: %d" % self.block_length),
                                  weight="heavy", size=self.text_size)
        self.text_sr = figtext(0.60, 0.88, ("Sample Rate: %.2f" % self.sample_rate),
                               weight="heavy", size=self.text_size)

        # figure init
        self.make_plots()

        # add button
        self.button_left_axes = self.fig.add_axes([0.45, 0.01, 0.05, 0.05], frameon=True)
        self.button_left = Button(self.button_left_axes, "<")
        self.button_left_callback = self.button_left.on_clicked(self.button_left_click)

        self.button_right_axes = self.fig.add_axes([0.50, 0.01, 0.05, 0.05], frameon=True)
        self.button_right = Button(self.button_right_axes, ">")
        self.button_right_callback = self.button_right.on_clicked(self.button_right_click)

        self.xlim = self.sp_iq.get_xlim()

        self.manager = get_current_fig_manager()
        connect('draw_event', self.zoom)
        connect('key_press_event', self.click)
        show()


    def make_plots(self):
        # if specified on the command-line, set file pointer
        self.hfile.seek(self.sizeof_data*self.start, 1)

        # Subplot for real and imaginary parts of signal
        self.sp_iq = self.fig.add_subplot(2,2,1, position=[0.075, 0.2, 0.4, 0.6])
        self.sp_iq.set_title(("I&Q"), fontsize=self.title_font_size, fontweight="bold")
        self.sp_iq.set_xlabel("Time (s)", fontsize=self.label_font_size, fontweight="bold")
        self.sp_iq.set_ylabel("Amplitude (V)", fontsize=self.label_font_size, fontweight="bold")

        # Subplot for Autocorrelation plot
        self.sp_ac = self.fig.add_subplot(2, 2, 2, position=[0.575, 0.2, 0.4, 0.6])
        self.sp_ac.set_title(("Autocorrelation Analysis"), fontsize=self.title_font_size, fontweight="bold")
        self.sp_ac.set_xlabel("Sample Index", fontsize=self.label_font_size, fontweight="bold")
        self.sp_ac.set_ylabel("Autocorrelation Coefficient", fontsize=self.label_font_size, fontweight="bold")

        self.get_data()

        self.plot_iq  = self.sp_iq.plot([], 'b-') # make plot for reals
        self.plot_iq += self.sp_iq.plot([], 'r-') # make plot for imags
        self.draw_time()                           # draw the plot

        self.plot_ac = self.sp_ac.plot([], 'b-')  # make plot for ac
        self.draw_ac()                              # draw the plot

        draw()

    def get_data(self):
        self.position = self.hfile.tell() / self.sizeof_data
        self.text_file_pos.set_text("File Position: %d" % (self.position))
        try:
            self.iq = np.fromfile(self.hfile, dtype=self.datatype, count=self.block_length)
        except MemoryError:
            print("End of File")
        else:
            # iq time sequences
            tstep = 1.0 / self.sample_rate
            # self.time = numpy.array([tstep*(self.position + i) for i in range(len(self.iq))])
            self.time = np.array([tstep * (i) for i in range(len(self.iq))])

            # calculate autocorrelation coefficient
            self.iq_ac = self.acf_norm(self.iq)
            self.sample_indices = np.arange(len(self.iq_ac))


    def moving_avg(self, data, window_size=48, scale=1.0):
        assert len(data) >= window_size, 'the length of data should not be smaller then nwindow'
        sum_list = []
        sum = np.sum(data[:window_size])
        sum_list.append(sum * scale)
        for i in range(1, len(data)):
            # print(data[i-1])
            if i + window_size - 1 < len(data):
                sum += data[i + window_size - 1]
                sum -= data[i - 1]
            else:
                sum -= data[i - 1]
            sum_list.append(sum * scale)
        return np.array(sum_list)

    def acf_norm(self, data, ndelay=16, nwindow=48):
        assert len(data) >= nwindow, 'the length of data should not be smaller then nwindow'

        # padding data with zeros
        data_padding = np.concatenate([data, np.zeros(ndelay)]).astype(np.complex64)
        padding_data = np.concatenate([np.zeros(ndelay), data]).astype(np.complex64)
        padding_data = np.conjugate(padding_data)

        # get the bigger amplitude between data_padding and padding_data elementwise
        data_bigger = np.max([np.abs(data_padding) ** 2, np.abs(padding_data) ** 2], axis=0)

        a_list = self.moving_avg(data_padding * padding_data, window_size=nwindow)
        p_list = self.moving_avg(data_bigger, window_size=nwindow)

        a_list = np.abs(a_list)

        return np.divide(a_list, p_list, out=np.zeros_like(a_list), where=p_list != 0)

    def draw_time(self):
        reals = self.iq.real
        imags = self.iq.imag
        self.plot_iq[0].set_data([self.time, reals])
        self.plot_iq[1].set_data([self.time, imags])
        self.sp_iq.set_xlim(self.time.min(), self.time.max())
        self.sp_iq.set_ylim([1.5*min([reals.min(), imags.min()]),
                             1.5*max([reals.max(), imags.max()])])

    def draw_ac(self):
        self.plot_ac[0].set_data([self.sample_indices, self.iq_ac])
        self.sp_ac.set_xlim(self.sample_indices.min(), self.sample_indices.max())
        self.sp_ac.set_ylim(-0.1, 1.1)

    def update_plots(self):
        self.draw_time()
        self.draw_ac()

        self.xlim = self.sp_iq.get_xlim()
        draw()

    def zoom(self, event):
        newxlim = np.array(self.sp_iq.get_xlim())
        curxlim = np.array(self.xlim)
        if (newxlim[0] != curxlim[0] or newxlim[1] != curxlim[1]):
            self.xlim = newxlim
            # xmin = max(0, int(ceil(self.sample_rate*(self.xlim[0] - self.position))))
            # xmax = min(int(ceil(self.sample_rate*(self.xlim[1] - self.position))), len(self.iq))
            xmin = max(0, int(ceil(self.sample_rate * (self.xlim[0]))))
            xmax = min(int(ceil(self.sample_rate * (self.xlim[1]))), len(self.iq))

            iq = self.iq[xmin: xmax]
            time = self.time[xmin: xmax]

            iq_ac = self.acf_norm(iq)
            sample_indices = np.arange(len(iq_ac))

            self.plot_ac[0].set_data(sample_indices, iq_ac)
            self.sp_ac.axis([sample_indices.min(), sample_indices.max(), -0.1, 1.1])

            draw()

    def click(self, event):
        forward_valid_keys = [" ", "down", "right"]
        backward_valid_keys = ["up", "left"]

        if (self.find(event.key, forward_valid_keys)):
            self.step_forward()

        elif (self.find(event.key, backward_valid_keys)):
            self.step_backward()

    def find(self, item_in, list_search):
        try:
            return list_search.index(item_in) != None
        except ValueError:
            return False

    def button_left_click(self, event):
        self.step_backward()

    def button_right_click(self, event):
        self.step_forward()

    def step_forward(self):
        self.get_data()
        self.update_plots()

    def step_backward(self):
        # Step back in file position
        if (self.hfile.tell() >= 2 * self.sizeof_data * self.block_length):
            self.hfile.seek(-2 * self.sizeof_data * self.block_length, 1)
        else:
            self.hfile.seek(-self.hfile.tell(), 1)
        self.get_data()
        self.update_plots()

    @staticmethod
    def set_options():
        description = "Takes a GNU Radio complex binary file and displays the I&Q data versus time as well as the Autocorrelation plot. The y-axis values are plotted assuming volts as the amplitude of the I&Q streams and Autocorrelation Coefficient (0-1). The script plots a certain block of data at a time, specified on the command line as -B or --block. This value defaults to 1000. The start position in the file can be set by specifying -s or --start and defaults to 0 (the start of the file). By default, the system assumes a sample rate of 1, so in time, each sample is plotted versus the sample number. To set a true time and frequency axis, set the sample rate (-R or --sample-rate) to the sample rate used when capturing the samples."

        parser = ArgumentParser(conflict_handler="resolve", description=description)
        parser.add_argument("-d", "--data-type", default="complex64",
                            choices=("complex64", "float32", "uint32", "int32", "uint16",
                                     "int16", "uint8", "int8"),
                            help="Specify the data type [default=%(default)r]")
        parser.add_argument("-B", "--block", type=int, default=40000,
                            help="Specify the block size [default=%(default)r]")
        parser.add_argument("-s", "--start", type=int, default=200000,
                            help="Specify where to start in the file [default=%(default)r]")
        parser.add_argument("-R", "--sample-rate", type=float, default=1.0,
                            help="Set the sampler rate of the data [default=%(default)r]")
        parser.add_argument("file", metavar="FILE",
                            help="Input file with samples")
        return parser




def main():
    parser = wifi_ana.set_options()
    args = parser.parse_args("-s 8000000 -B 100000 data/iq_20M_g.dat".split())
    # args = parser.parse_args("-s 614500 -B 6500 data/iq_20M_g.dat".split()) # maybe bluebooth or zigbee signal, ac < 0.2
    # args = parser.parse_args("-s 1953000 -B 10000 data/iq_20M_g.dat".split()) # maybe bluebooth or zigbee signal, ac < 0.2
    # args = parser.parse_args("-s 2409500 -B 10000 data/iq_20M_g.dat".split()) # maybe bluebooth or zigbee signal, ac < 0.2
    # args = parser.parse_args("-s 2845600 -B 1000 data/iq_20M_g.dat".split()) # some signal occur, ac > 0.6
    # args = parser.parse_args("-s 2875000 data/iq_20M_g.dat".split()) # some signal occur, ac < 0.6
    # args = parser.parse_args("-s 3655000 data/iq_20M_g.dat".split()) # some signal occur, ac < 0.6
    # args = parser.parse_args("-s 3835000 data/iq_20M_g.dat".split()) # some short signal occur, ac < 0.6
    # args = parser.parse_args("-s 4065000 data/iq_20M_g.dat".split()) # some short signal occur, ac > 0.6
    # args = parser.parse_args("-s 4830000 data/iq_20M_g.dat".split()) # some short signal occur, ac < 0.6
    # args = parser.parse_args("-s 6405000 data/iq_20M_g.dat".split()) # some short signal occur, ac < 0.6
    args = parser.parse_args("-s 6542700 -B 700 data/iq_20M_g.dat".split()) # wifi occur, ac > 0.9
    # args = parser.parse_args("-s 6840000 data/iq_20M_g.dat".split()) # some short signal occur, ac < 0.6
    # args = parser.parse_args("-s 7205000 data/iq_20M_g.dat".split()) # some short signal occur, ac < 0.6
    # args = parser.parse_args("-s 7460000 data/iq_20M_g.dat".split()) # some short signal occur, ac > 0.5
    # args = parser.parse_args("-s 7510000 data/iq_20M_g.dat".split()) # some signal occur, ac > 0.6
    # args = parser.parse_args("-s 7860000 data/iq_20M_g.dat".split()) # some signal occur, ac < 0.2
    # args = parser.parse_args("-s 8055000 data/iq_20M_g.dat".split()) # some signal occur, ac > 0.6
    # args = parser.parse_args("-s 8485000 data/iq_20M_g.dat".split()) # some signal occur, ac < 0.3
    wifi_ana(args.file, args)

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass