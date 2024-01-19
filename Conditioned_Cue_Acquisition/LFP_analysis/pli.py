import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
from numpy.fft import fft, ifft
from lfp import load_data, compute_mwt, pli

parser = argparse.ArgumentParser(description='loads in dict of trialized reocording from two regions and computes phase lag index between them')
parser.add_argument('path_r1', metavar='<path>', help='path of curated data to load from region 1')
parser.add_argument('path_r2', metavar='<path>', help='path of curated data to load from region 2')
parser.add_argument('f_low', type=int, help='low cut of mwt')
parser.add_argument('f_high', type=int, help='high cut of mwt')
parser.add_argument('-nlow', '--nlow', action='store_true', help='construct n dic for lower freqs during mwt')
args = parser.parse_args()

def plot_pli(pli, t_zero, freqs, label):
    f, ax = plt.subplots(1,1)
    f.set_figheight(4)
    f.set_figwidth(7)

    fs = 2000
    time = np.arange((0-t_zero) / fs, (np.shape(pli)[1] - t_zero) / fs, 1/fs)
    title = ' '.join(label.split('_'))

    n = 50
    vmin = 0   #================== harded coded params =================#
    vmax = 0.6
    levels = np.linspace(vmin, vmax, n+1)

    # plot contour fill
    cs = ax.contourf(time, freqs, pli, levels=levels, cmap='jet')
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
    f.savefig(' '.join([title, frange,'PLI.png']), bbox_inches='tight', dpi=600)
    #np.save(' '.join([title, frange,'Spectrogram.npy']), pli, allow_pickle=True)

    # plt.show()
    # plt.close()

def main():
    # set up
    label_r1, data_r1 = load_data(args.path_r1)
    label_r2, data_r2 = load_data(args.path_r2)

    r1 = label_r1.split('_')[-1]
    r2 = label_r2.split('_')[-1]
    label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

    freqs = np.arange(args.f_low, args.f_high+1)
    fs = 2000
    b_i = 1000  # 500 ms baseline period right before plot
    t_0 = 600   # 4000 sample epochs sliced at [400:3400] after mwt
                # result is baseline=[400:1400], plot=[1400:3400], t=0 @ 2000
                # or in terms of mwt slice: [0:1000], [1000:3000], t=0 @ 1600; therefore b_i=1000
                # or in terms of plot slice: [-1000:0], [0:2000], t=0 @ 600; therefore t_0=600
    if args.nlow:
        ns = np.linspace(3,7, len(freqs))               # ns for low freq
    else:
        ns = np.linspace(6,12, len(freqs))              # ns for high freq
    n_dict = dict(zip(freqs, ns))

    # depending on the version of python, dicts may be unordered
    # therefore, rather than tossing data from both regions into a data matrix,
    # I can simply access them pairwise by key (all data dicts share keys across
    # regions)
    assert set(data_r1.keys()) == set(data_r2.keys())
    trial_ns = list(data_r1.keys())

    # compute mwt, extract phase, slice out section of interest
    pha_trials_r1 = []
    for trial_n in data_r1:
        sig = data_r1[trial_n]
        pha = [np.angle(compute_mwt(sig, fs, f, n_dict[f])) for f in freqs]
        pha = [f_pha[400+b_i:3400] for f_pha in pha]     # cutting off edges
        pha_trials_r1.append(pha)

    # compute mwt, extract phase, slice out section of interest
    pha_trials_r2 = []
    for trial_n in data_r2:
        sig = data_r2[trial_n]
        pha = [np.angle(compute_mwt(sig, fs, f, n_dict[f])) for f in freqs]
        pha = [f_pha[400+b_i:3400] for f_pha in pha]     # cutting off edges
        pha_trials_r2.append(pha)

    # compute ispc across samples for every trial (for every f)
    plis = pli(pha_trials_r1, pha_trials_r2)

    # plot
    plot_pli(plis, t_0, freqs, label)

    sys.exit()

if __name__ == '__main__':
    main()
