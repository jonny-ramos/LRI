import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
from scipy import stats
from numpy.fft import fft, ifft
from scipy.signal import butter, filtfilt, hilbert
from lfp import load_data, butter_bandpass, butter_bandpass_filter, filter_hilbert

'''
takes trialized signal timeseries data from recorded from two brain regions and
first, computes instantaneous power envelope via filter-hilbert method for
specified frequency range (here, theta from 4-12 Hz). Envelopes are concatenated
and then, in a sliding window fashion, computes correlation (spearman's rho)
between the two envelopes across varying lags (here, ranging from -100 ms to
+100 ms). Plots trace of correlation coefficients over lags, indicating time at
which peak correlation occurs. The convention in the current work is that a
negative lag indicates that R2 leads and a positive lag indicates that R1 leads.
'''

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data from two regions and plots mean power by binned phase')
parser.add_argument('path_r1', metavar='<path>', help='path of curated data to load from region 1, r1')
parser.add_argument('path_r2', metavar='<path>', help='path of curated data to load from region 2, r2')
parser.add_argument('f_pwr', help='comma separated ints denoting high and low cut f for power extraction')
args = parser.parse_args()

def plot_lag_xcorr(xcorr_twindows, title):
    xcorr_R = [r for r, p in xcorr_twindows]
    lags = np.linspace(-100,100,400)

    plt.figure(figsize=(6,4))
    plt.ylabel('Correlation')
    plt.xlabel('Lag (ms)')
    plt.axvline(x=0, color='k', ls='--', alpha=0.5)
    plt.title(title)

    if 'ChABC' in title:
        plt.plot(lags, xcorr_R, 'r', zorder=1)
    else:
        plt.plot(lags, xcorr_R, 'k', zorder=1)

    plt.scatter(lags[xcorr_R.index(np.max(xcorr_R))], np.max(xcorr_R), s=8**2,
    marker=(5,1), zorder=2, color='yellow', edgecolors='k', label=f'R={np.round(np.max(xcorr_R), 2)}\nat {np.round(lags[xcorr_R.index(np.max(xcorr_R))], 2)} ms')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size':6})
    plt.tight_layout()
    plt.savefig(title+'.png', dpi=600)

def main():
    label_r1, data_r1 = load_data(args.path_r1)
    label_r2, data_r2 = load_data(args.path_r2)

    r1 = label_r1.split('_')[-1]
    r2 = label_r2.split('_')[-1]
    label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

    f_low, f_high = tuple([int(n) for n in args.f_pwr.split(',')])


    fs = 2000
    b_i = 1000
    t_0 = 600

    assert data_r1.keys() == data_r2.keys()
    trial_ns = list(set(data_r1.keys()))
    n = len(trial_ns)

    R1_pwr = [np.abs(filter_hilbert(data_r1[trial_n], f_low, f_high, fs, 4)) for trial_n in trial_ns]
    R2_pwr = [np.abs(filter_hilbert(data_r2[trial_n], f_low, f_high, fs, 4)) for trial_n in trial_ns]

    R1_pwr = [trial - np.mean(trial) for trial in R1_pwr]
    R2_pwr = [trial - np.mean(trial) for trial in R2_pwr]

    # the stationary timeseries
    R1_amp = np.concatenate([trial[2200:3800] for trial in R1_pwr])

    # the sliding timeseries
    R2_amp = [trial[2000:4000] for trial in R2_pwr]

    # generating sliding R2_timeseries
    xcorr_twindows = []
    for i in range(400):    # 400 because that encompasses -100 to +100 ms sliding time window
        R2_wind = np.concatenate([trial[0+i: 1600+i] for trial in R2_amp])   # cncat and caclcualte r for all trials at once

        assert np.shape(R1_amp) == np.shape(R2_wind)

        r, p = stats.spearmanr(R1_amp, R2_wind)
        xcorr_twindows.append((r, p))

    title = label + '_theta_lag_xcorr'
    plot_lag_xcorr(xcorr_twindows, title)

if __name__ == '__main__':
    main()
