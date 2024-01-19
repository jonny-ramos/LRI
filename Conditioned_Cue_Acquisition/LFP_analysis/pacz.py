import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
import itertools
import multiprocessing
import datetime
from numpy.fft import fft, ifft
from scipy.signal import butter, filtfilt, hilbert
from lfp import load_data, compute_mwt, baseline_norm



parser = argparse.ArgumentParser(description='loads in dict of trialized recording data from two regions and plots mean power by binned phase')
parser.add_argument('path_r1', metavar='<path>', help='path of curated data to load from region 1, r1')
parser.add_argument('path_r2', metavar='<path>', help='path of curated data to load from region 2, r2')
parser.add_argument('f_pha', help='comma separated ints denoting high and low cut f for phase extraction')
parser.add_argument('f_pwr', help='comma separated ints denoting high and low cut f for power extraction')
args = parser.parse_args()

def mwt_pha(data, trial_ns, f_pha, n_dict_pha, b_i, t_0):
    '''
    takes dict of signals and computes instantaneous phase angles at a single
    peak frequency via mwt
    args:
        data: dict(t), t: the number of trials, dict values are np.ndarray(N)
            N: the number of samples (timepoints)
        trial_ns: list(t), dict keys for accessing signal arrays from data dict;
            t: the number of trials
        f_pha: numpy.float64, peak frequency of morlet wavelet
        n_dict_pha: dict(p), maps frequency to number of cycles for mwt;
            a: the number of frequencies for which amplitude is computed;
            for each key:val pair, key: numpy.float64; val: numpy.float64
        t_0: int, index at time = 0; data before t_0 will be sliced out
    return:
        pha: np.ndarray(t, p, N); matrix containing instantaneous phase angles
            t: the number of trials; p: the number of frequencies for which
            phase was computed; N: the number of samples
    '''
    pha = []
    for trial in trial_ns:
        asig = compute_mwt(data[trial], fs=2000, peak_freq=f_pha, n=n_dict_pha[f_pha])
        phase = np.angle(asig[500:-500])
        assert len(phase) == 3000

        post_press = phase[b_i+t_0:]
        assert len(post_press) == 1500

        pha.append(post_press)

    return np.array(pha)

def mwt_amp(data, trial_ns, f_amp, n_dict_amp, b_i, t_0, norm=True):
    '''
    takes a dict of signals are computes instantaneous amplitude at a single
    peak frequency via mwt
    args:
        data: dict(t), t: the number of trials, dict values are np.ndarray(N)
            N: the number of samples (timepoints)
        trial_ns: list(t), dict keys for accessing signal arrays from data dict;
            t: the number of trials
        f_amp: numpy.float64, peak frequency of morlet wavelet
        n_dict_amp: dict(a), maps frequency to number of cycles for mwt;
            a: the number of frequencies for which amplitude is computed;
            for each key:val pair, key: numpy.float64; val: numpy.float64
        t_0: int, index at time = 0; data before t_0 will be sliced out
    return:
        amp: np.ndarray(t, a, N); 3D matrix containing instantaneous amplitude
            t: the number of trials; a: the number of frequencies for which
            amplitude was computed; N: the number of samples
    '''
    def _baseline_norm(raw_pwr, bline_index, db=False):
        '''
        small baseline fn for just a single frequency of raw_pwr, that is,
        raw_pwr is 1D
        args:
            raw_pwr(N): np.ndarray, N: number of samples
            bline_index: int, denotes index where trial begins and baseline ends
        return:
            norm_pwr: np.npdarray(t,N-bline_index), note section of data considered
                as baseline is not returned
        '''
        sig = raw_pwr[bline_index:]
        baseline = raw_pwr[:bline_index]
        b_mean = np.average(baseline)

        if not db:
            norm_sig = np.array(sig) / b_mean
        elif db:
            norm_sig = 10 * np.log10(sig / b_mean)

        return norm_sig

    amp = []
    for trial in trial_ns:
        asig = compute_mwt(data[trial], fs=2000, peak_freq=f_amp, n=n_dict_amp[f_amp])

        if norm:
            pwr = np.abs(asig[500:-500])**2
            assert len(pwr) == 3000

            nrm_pwr = _baseline_norm(pwr, b_i)
            assert len(nrm_pwr) == 2000

            post_press = nrm_pwr[t_0:]
            assert len(post_press) == 1500

        else:
            pwr = np.abs(asig[500:-500])**2
            assert len(pwr) == 3000

            post_press = pwr[b_i+t_0:]
            assert len(post_press) == 1500

        amp.append(post_press)

    return np.array(amp)

def pac(pha, amp):
    '''
    compute mean vector length from instantaneous phase and amplitude timeseries
    args:
        pha: np.ndarray(N), N: the number of samples (timepoints)
        amp: np.ndarray(N), N: the number of samples (timepoints)
    return:
        pac: float, the computed mean vector length
    '''
    return np.abs(np.mean(amp * np.exp(1j * pha)))

def timeshift(trials):
    '''
    takes a 2D array of signal data and for each trial, shifts data by a random
    number of indices by slicing the trial at a random index, and then switching
    the order of the two slices
    args:
        trials: np.ndarray(t, N), t: the number of trials; N the number of samples
    return:
        shifted: np.ndarray(t, N), t: the number of trials; N the number of samples
    '''
    # draw t number of random integers (indices) from uniform distribution ranging from 1 to N
    slice_i = np.random.uniform(1,np.shape(trials)[1], np.shape(trials)[0]).astype(int)

    # slice each trial at random index, concat back together in reverse order
    shifted = np.array([np.concatenate([trial[i:],trial[:i]]) for trial, i in zip(trials, slice_i)])

    return shifted

def zscore_pac(pha, amp, observed_pac, n_iter):
    '''
    takes instantaneous phase and instantaneous amplitude, generates null (H0)
    distribution of phase amplitude coupling (pac) mean vector length (mvl),
    and standardizes observed pac to null distribution.
    args:
        pha: np.ndarray(t, N), t: the number of trials; N the number of samples
        amp: np.ndarray(t, N), t: the number of trials; N the number of samples
        observed_pac: numpy.float64, raw pac value to z-standardize
        n_iter: int, the number of iterations to construct H0 with
    return:
        pacz: numpy.float64, the z-standardized pac mvl
    '''
    # no need to shuffle both timeseries
    pha_h0 = np.concatenate(pha)

    # generate null distribution
    h0_distribution = [None] * n_iter

    for i in range(n_iter):
        amp_h0 = np.concatenate(timeshift(amp))
        pac_h0 = pac(pha_h0, amp_h0)
        h0_distribution[i] = pac_h0

    # standardize observed pac to null distribution
    pacz = (observed_pac - np.mean(h0_distribution)) / np.std(h0_distribution)

    return pacz

def wrapper(fpair, time=True):
    '''
    wrapper function for permutation testing optimized for multiprocessing
    with pool. 
    '''

    if time:
        start = datetime.datetime.now()

    f_pha, f_amp = fpair
    pha = pha_dict[f_pha]
    amp = amp_dict[f_amp]
    observed_pac = pac(np.concatenate(pha), np.concatenate(amp))
    pacz = zscore_pac(pha, amp, observed_pac, n_iter)

    if time:
        end = datetime.datetime.now()
        print(f'{fpair}: {end-start}')

    return ((f_pha, f_amp), pacz)

def plot_pac(pac_cm, freqs4pha, freqs4amp, label):
    floor = np.min(pac_cm)
    ceil = np.max(pac_cm)

    f, ax = plt.subplots(1,1)
    f.set_figheight(5)
    f.set_figwidth(5)

    #cs = ax.contourf(freqs4pha, freqs4amp, pac_cm, levels=np.geomspace(floor, ceil), extend='min', cmap='jet')
    cs = ax.contourf(freqs4pha, freqs4amp, pac_cm, levels=np.linspace(floor, ceil), extend='min', cmap='jet')
    cbar = plt.colorbar(cs, shrink=0.9)
    cbar.ax.tick_params(labelsize=20)

    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xlabel('Frequency for phase (Hz)', fontsize=22)
    ax.set_ylabel('Frequency for power (Hz)', fontsize=22)
    ax.set_title(label, size=10)

    plt.setp(ax, xticks=[4,6,8,10,12], xticklabels=['4','6','8','10','12'])
    plt.tight_layout()
    plt.savefig(label+'_RAW_PAC_POWER_NORM_zscoretest.png', dpi=600)

# this main function is how I would do this WITHOUT multiprocessing
# then just call main if __name__ == '__main__' and spit out result file
def main():
    # load data
    label_r1, data_r1 = load_data(args.path_r1)
    label_r2, data_r2 = load_data(args.path_r2)

    # some str manipulation to generate label
    r1 = label_r1.split('_')[-1]
    r2 = label_r2.split('_')[-1]
    label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

    # from args, generate arrays of frequencies to be used for PAC
    f_pha_low, f_pha_high = tuple([int(n) for n in args.f_pha.split(',')])
    f_amp_low, f_amp_high = tuple([int(n) for n in args.f_pwr.split(',')])
    freqs4pha = np.linspace(f_pha_low, f_pha_high, (f_pha_high - f_pha_low)*2 + 1)
    freqs4amp = np.linspace(f_amp_low, f_amp_high, (f_amp_high - f_amp_low) + 1)

    # set up
    fs = 2000
    b_i = 1000
    t_0 = 500
    n_iter=1000

    # check that keys match, we will be acceessing trials by key
    assert set(data_r1.keys()) == set(data_r2.keys())
    trial_ns = sorted(list(data_r1.keys()))

    ns_pha = dict(zip(freqs4pha, np.linspace(3, 6, len(freqs4pha))))
    ns_amp = dict(zip(freqs4amp, np.linspace(6, 12, len(freqs4amp))))

    # init empty containers
    val = []
    pha_dict = dict()
    amp_dict = dict()

    start = datetime.datetime.now()

    for f_pha, f_amp in itertools.product(freqs4pha, freqs4amp):
        # check if we've already computed this previously
        if not f_pha in pha_dict:
            pha = mwt_pha(data_r2, trial_ns, f_pha, ns_pha, b_i, t_0)
            pha_dict.update({f_pha: pha})
        else:
            pha = pha_dict[f_pha]

        # check if we've already computed this previously
        if not f_amp in amp_dict:
            amp = mwt_amp(data_r1, trial_ns, f_amp, ns_amp, b_i, t_0)
            amp_dict.update({f_amp: amp})
        else:
            amp = amp_dict[f_amp]

        observed_pac = pac(np.concatenate(pha), np.concatenate(amp))
        pacz = zscore_pac(pha, amp, observed_pac, n_iter)

        val.append(pacz)

    cm = np.reshape(val, (len(freqs4pha), len(freqs4amp))).T
    print(np.shape(val))

    end = datetime.datetime.now()
    print(f'single processor: {end-start}')

    sys.exit()

    # write to binary file in case we want to adjust the plot later
    np.save(label+'.npy', cm, allow_pickle=True)

    # time to plot!
    plot_pac(cm, freqs4pha, freqs4amp, label)

### However, because multiprocessing pool serializes with pickle, we must
### construct dicts containing composite signal globally (NOT in main function)
### (local objects are not pickleable)
# load data
label_r1, data_r1 = load_data(args.path_r1)
label_r2, data_r2 = load_data(args.path_r2)

# some str manipulation to generate label
r1 = label_r1.split('_')[-1]
r2 = label_r2.split('_')[-1]
label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

# from args, generate arrays of frequencies to be used for PAC
f_pha_low, f_pha_high = tuple([int(n) for n in args.f_pha.split(',')])
f_amp_low, f_amp_high = tuple([int(n) for n in args.f_pwr.split(',')])
freqs4pha = np.linspace(f_pha_low, f_pha_high, (f_pha_high - f_pha_low)*2 + 1)
freqs4amp = np.linspace(f_amp_low, f_amp_high, (f_amp_high - f_amp_low) + 1)

# set up
fs = 2000
b_i = 1000
t_0 = 500
n_iter=10000

# check that keys match, we will be acceessing trials by key
assert set(data_r1.keys()) == set(data_r2.keys())
trial_ns = sorted(list(data_r1.keys()))

ns_pha = dict(zip(freqs4pha, np.linspace(3, 6, len(freqs4pha))))
ns_amp = dict(zip(freqs4amp, np.linspace(6, 12, len(freqs4amp))))

# init empty containers
cm = []
pha_dict = dict()
amp_dict = dict()

# construct up front global dict containing pha/amp (2D) for every pair of freqs
for f_pha, f_amp in itertools.product(freqs4pha, freqs4amp):
    # check if we've already computed this previously
    if not f_pha in pha_dict:
        pha = mwt_pha(data_r2, trial_ns, f_pha, ns_pha, b_i, t_0)
        pha_dict.update({f_pha: pha})
    else:
        pha = pha_dict[f_pha]

    # check if we've already computed this previously
    if not f_amp in amp_dict:
        amp = mwt_amp(data_r1, trial_ns, f_amp, ns_amp, b_i, t_0)
        amp_dict.update({f_amp: amp})
    else:
        amp = amp_dict[f_amp]

# multiprocessing step is carried out in main
if __name__ == '__main__':

    start = datetime.datetime.now()

    # pool takes list of args, I want every pair in the cartesian product of the two arrays
    fpairs = list(itertools.product(freqs4pha, freqs4amp))

    # init multiprocessing pool
    pool = multiprocessing.Pool(processes=8)

    # map wrapper fn and args to pool (this takes a while)
    result = pool.map(wrapper, fpairs)

    # sort by fpair key (wrapper returns results in (key, val) format)
    result.sort(key=lambda x: x[0])

    # accessing the value we want from result
    val = list(map(lambda x: x[1], result))

    # reshape for plotting; transpose to plot frequncy for phase on the x axis
    cm = np.reshape(val, (len(freqs4pha), len(freqs4amp))).T

    end = datetime.datetime.now()
    print(f'multiprocessing pool: {end-start}')

    # write to binary file in case we want to adjust the plot later
    np.save(label+'_pacz.npy', cm, allow_pickle=True)

    # time to plot!
    plot_pac(cm, freqs4pha, freqs4amp, label)
