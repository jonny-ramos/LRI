import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
from numpy.fft import fft, ifft
from lfp import load_data, compute_mwt, baseline_norm

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data')
parser.add_argument('path', metavar='<path>', help='path of curated data to load')
parser.add_argument('f_low', type=int, help='low cut of mwt')
parser.add_argument('f_high', type=int, help='high cut of mwt')
parser.add_argument('-nlow', '--nlow', action='store_true', help='construct n dict for lower freqs during mwt')
args = parser.parse_args()

# plot
def plot_spectrogram(norm_avg, t_zero, freqs, label):
    f, ax = plt.subplots(1,1)
    f.set_figheight(4)
    f.set_figwidth(7)

    fs = 2000
    time = np.arange((0-t_zero) / fs, (np.shape(norm_avg)[1] - t_zero) / fs, 1/fs)
    title = ' '.join(label.split('_'))

    n = 50
    vmin = np.min(norm_avg)    #================== harded coded params =================#
    vmax = np.max(norm_avg)
    levels = np.linspace(vmin, vmax, n+1)

    # plot contour fill
    cs = ax.contourf(time, freqs, norm_avg, levels=levels, cmap='magma')
    ax.set_title(title, size=16)
    ax.set_xlabel('Time (ms), press @ t=0', fontsize=16)
    ax.set_ylabel('Frequency (Hz)', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # add color bar
    cb = f.colorbar(cs, ax=ax, shrink=0.9, ticks=np.linspace(vmin,vmax, 7))
    cb.ax.tick_params(labelsize=16)
    cb.ax.set_yticklabels(np.round(np.linspace(0,vmax, 7),1))

    frange = f'{freqs.min()}-{freqs.max()} Hz'
    f.savefig(' '.join([title, frange,'Spectrogram.png']), bbox_inches='tight', dpi=600)
    # np.save(' '.join([title, frange,'Spectrogram.npy']), norm_avg, allow_pickle=True)

    # plt.show()
    # plt.close()

def main():
    # set up
    label, data = load_data(args.path)
    freqs = np.arange(args.f_low, args.f_high+1)
    fs = 2000
    b_i = 1000  # 500 ms baseline period right before plot
    t_0 = 600   # 4000 sample epochs sliced at [400:3400] after mwt
                # result is baseline=[400:1400], plot=[1400:3400], t=0 @ 2000
                # or in terms of mwt slice: [0:1000], [1000:3000], t=0 @ 1600; therefore b_i=1000
                # or in terms of plot slice: [-1000:0], [0:2000], t=0 @ 600; therefore t_0=600
    if args.nlow:
        ns = np.linspace(3,6, len(freqs))               # ns for low freq
    else:
        ns = np.linspace(6,12, len(freqs))              # ns for high freq
    n_dict = dict(zip(freqs, ns))

    # take data out of dict and toss into data matrix, npndarray(L,N): where
    # L: number of trials, N: number of samples (time points)
    data = np.array(list(data.values()))

    # compute mwt, extract power, normalize per trial
    norm_trials = []
    for sig in data:
        raw_pwr = [abs(compute_mwt(sig, fs, f, n_dict[f]))**2 for f in freqs]
        raw_pwr = [f_pwr[400:3400] for f_pwr in raw_pwr ] # cut off edges we don't want to plot
        norm_pwr =baseline_norm(raw_pwr, b_i, db=False)
        norm_trials.append(norm_pwr)

    # average across trials
    norm_avg = np.mean(norm_trials, axis=0)

    # plot
    plot_spectrogram(norm_avg, t_0, freqs, label)

    sys.exit()

if __name__ == '__main__':
    main()
