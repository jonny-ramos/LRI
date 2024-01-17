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
specified frequency range (here, theta from 4-12 Hz). Then for each trial (NOT
concatenated) in a sliding window fashion, computes correlation (spearman's rho)
between the two envelopes across varying lags (here, ranging from -100 ms to
+100 ms). Plots peak correlation heat map for each trial. The convention in the
current work is that a negative lag indicates that R2 (sliding window) leads
and a positive lag indicates that R1 (stationary window) leads.
'''

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data from two regions and plots mean power by binned phase')
parser.add_argument('path_r1', metavar='<path>', help='path of curated data to load from region 1, r1')
parser.add_argument('path_r2', metavar='<path>', help='path of curated data to load from region 2, r2')
parser.add_argument('f_pwr', help='comma separated ints denoting high and low cut f for power extraction')
args = parser.parse_args()

def lag_xcorr_bytrial(data_r1, data_r2, trial_ns, f_low, f_high, order):
    fs=2000
    R1_pwr = [abs(filter_hilbert(data_r1[trial_n], f_low, f_high, fs, 4))**2 for trial_n in trial_ns]
    R2_pwr = [abs(filter_hilbert(data_r2[trial_n], f_low, f_high, fs, 4))**2 for trial_n in trial_ns]

    # the stationary timeseries
    R1_amp = [trial[2200:3800] for trial in R1_pwr]

    # the sliding timeseries
    R2_amp = [trial[2000:4000] for trial in R2_pwr]

    xcorr_bytrial = []
    for R1_trial, R2_trial in zip(R1_amp, R2_amp):

        xcorr_trial  = []
        for i in range(400):

            R2_wind = R2_trial[0+i: 1600+i]

            assert np.shape(R1_trial) == np.shape(R2_wind)

            r, p = stats.spearmanr(R1_trial, R2_wind)

            xcorr_trial.append(r)

        xcorr_bytrial.append(xcorr_trial)

    return xcorr_bytrial

def plot_xcorr_bytrial(xcorr_bytrial, title):
    def _sort_trials_by_max(unsorted_bytrial):
        iimax = []
        for i, arr in enumerate(unsorted_bytrial):
            i_max = np.where(arr == np.max(arr))[0].item()
            iimax.append((arr, i_max))

        arr_imax_sort = sorted(iimax, key=lambda x: x[1])
        arr_sortbymax = np.array([arr for arr, i in arr_imax_sort])

        return arr_sortbymax

    def _norm_minmax(raw_xcorr_bytrial):
        norm = [(arr - np.min(arr)) / (np.max(arr) - np.min(arr)) for arr in raw_xcorr_bytrial]

        return np.array(norm)

    xcorr_bytrial = _sort_trials_by_max(xcorr_bytrial)
    xcorr_bytrial = _norm_minmax(xcorr_bytrial)

    f, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(xcorr_bytrial, cmap='jet', aspect='auto', vmin=0, vmax=1, interpolation='nearest')

    lag_ticks = np.linspace(0,400,9).tolist()
    lags = np.linspace(-100,100,9).astype(int)
    ax.set_xticks(lag_ticks)
    ax.set_xticklabels(lags)
    ax.axes.axvline(200,c='k',ls='--')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('trial')
    ax.set_title(title)

    plt.savefig(title.replace('\n', ' ')+'.png', dpi=600)

def main():
    label_r1, data_r1 = load_data(args.path_r1)
    label_r2, data_r2 = load_data(args.path_r2)

    r1 = label_r1.split('_')[-1]
    r2 = label_r2.split('_')[-1]
    label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

    f_low, f_high = tuple([int(n) for n in args.f_pwr.split(',')])
    title = label + '_theta_lag_xcorr'

    fs = 2000
    b_i = 1000
    t_0 = 600

    assert data_r1.keys() == data_r2.keys()
    trial_ns = set(data_r1.keys())
    n = len(trial_ns)

    xcorr_bytrial = lag_xcorr_bytrial(data_r1, data_r1, trial_ns, f_low, f_high, 4)
    plot_xcorr_bytrial(xcorr_bytrial, title)

    sys.exit()

if __name__ == '__main__':
    main()
