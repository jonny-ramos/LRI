import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
from numpy.fft import fft, ifft
from scipy.signal import butter, filtfilt, hilbert
from lfp import load_data, baseline_norm, filter_hilbert, bin_pha, modulation_index

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data from two regions and plots mean power by binned phase')
parser.add_argument('path_r1', metavar='<path>', help='path of curated data to load from region 1, r1')
parser.add_argument('path_r2', metavar='<path>', help='path of curated data to load from region 2, r2')
parser.add_argument('f_pha', help='comma separated ints denoting high and low cut f for phase extraction')
parser.add_argument('f_pwr', help='comma separated ints denoting high and low cut f for power extraction')
args = parser.parse_args()


def plot_binned_pwr_dist(mean_binned_pwr, mi, label, t_range):
    if 'ChABC' in label:
        c='red'
    elif 'Vehicle' in label:
        c='darkgray'

    f_pha_low, f_pha_high = tuple([int(n) for n in args.f_pha.split(',')])
    f_pwr_low, f_pwr_high = tuple([int(n) for n in args.f_pwr.split(',')])


    pi = np.pi
    x = np.linspace(-pi, pi, 30)

    f = plt.figure(figsize=(6,5))
    plt.bar(x, mean_binned_pwr, width=pi/15, color=c, edgecolor='k')
    plt.xticks(np.arange(-pi, pi+pi/2, step=(pi/2)), ['-π','-π/2','0','π/2','π'])
    plt.xlabel(f'Theta Phase, {f_pha_low}-{f_pha_high} Hz, (rad)', fontsize=20)
    plt.ylabel(f'Relative Gamma Power,\n {f_pwr_low}-{f_pwr_high} Hz', fontsize=20)
    #plt.ylim(0,1.6)
    plt.title(f'{label}\n MI={np.round(mi,10)}, {t_range[0]}-{t_range[1]}ms')
    f.axes[0].tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(f'{label} MI {t_range[0]}-{t_range[1]}ms.png',dpi=600)

def main():
    label_r1, data_r1 = load_data(args.path_r1)
    label_r2, data_r2 = load_data(args.path_r2)

    r1 = label_r1.split('_')[-1]
    r2 = label_r2.split('_')[-1]
    label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

    f_pha_low, f_pha_high = tuple([int(n) for n in args.f_pha.split(',')])
    f_pwr_low, f_pwr_high = tuple([int(n) for n in args.f_pwr.split(',')])

    fs = 2000
    b_i = 1000
    t_0 = 600

    assert data_r1.keys() == data_r2.keys()
    trial_ns = set(data_r1.keys())
    n = len(trial_ns)

    trial_len = 1400
    pwr_r1 = np.zeros(shape=(n, trial_len))
    pha_r2 = np.zeros(shape=(n, trial_len))
    t_range = (0, int(trial_len / fs * 1000))

    for i, trial_n in enumerate(trial_ns):
        # gamma power time series
        sig = data_r1[trial_n]
        raw_pwr = abs(filter_hilbert(sig, f_pwr_low, f_pwr_high, fs, 6))**2
        raw_pwr = np.array([raw_pwr[400:3400]]) # cut out baseline
        norm_pwr = np.squeeze(baseline_norm(raw_pwr, b_i))
        norm_pwr = norm_pwr[t_0:t_0+trial_len] # only keep data after t=0
        pwr_r1[i] = norm_pwr

        # theta phase time series
        sig = data_r2[trial_n]
        pha = np.angle(filter_hilbert(sig, f_pha_low, f_pha_high, fs, 2))
        pha = pha[400:3400]
        pha = pha[b_i+t_0:b_i+t_0+trial_len]
        pha_r2[i] = pha

    # concat trials into one long array
    pwr = pwr_r1.flatten()
    pha = pha_r2.flatten()

    # binning instantaneous phase angles
    pha_bins = bin_pha(pha)

    # compute mean pwr per phase bin
    binned_pwr_means = [np.mean(pwr[bin]) for bin in pha_bins]

    # compute modulation index
    mi = modulation_index(binned_pwr_means, pha_bins)

    # plot
    plot_binned_pwr_dist(binned_pwr_means, mi, label, t_range)

    sys.exit()

if __name__ == '__main__':
    main()
