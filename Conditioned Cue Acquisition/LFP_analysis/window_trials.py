'''
Jonathan Ramos
11/11/2023
In this script am taking the raw epochs of data (+/- 1 second around event timestamp)
and computing average power across several different frequency bands via morlet wavelet convolution
since I have data from two brain areas, I will also include some standard measures
of inter-regional coherence between the prefrontal cortex and the hippocampus:
intersite phase coupling (a measure of how aligned phases angles are)
and canolty et al's MLV or mean vector length as a measure of phase ampltiude coupling.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from numpy.fft import fft, ifft

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data from two regions and computes intersite phase coupling between them')
parser.add_argument('path_r1', metavar='<path>', help='path of curated data to load')
parser.add_argument('path_r2', metavar='<path>', help='path of curated data to load')
args = parser.parse_args()

def load_data(file):
    '''
    loads in data file. returns label and dict of curated data.
    '''
    # files = glob.glob('/Users/jonathanramos/Desktop/CCA NEW ANALYSES/curation_2023/data_curated/CUE0/*.npy')
    f = file
    label = '_'.join(f.replace('_curated.npy','').split('/')[-2:])
    data = np.load(f, allow_pickle=True)

    return label, data.item()

def compute_mwt(signal, fs, peak_freq, n):
    '''
    Takes a timeseries and computes morlet wavelet convolution.
    First, generates a wavelet of peak frequency = peak_freq, then takes fft of signal and fft of
    wavelet. Then takes ift of the product of fft of signal and fft of wavelet and cuts off the
    "wings" of convolution (1/2 length wavelet from either end). I know the length of our wavelet
    is an odd number of points (fs=2000) so it's more convenient to use floor division here.
    Lastly, extracts amplitude and phase from the result of convolution.

    args:
        signal: list, timeseries of volatage readings (floats)
        fs: int, sampling rate of timeseries data in hz
        peak_freq: int, peak freq of wavelet
        n: int, number of cycles to construct wavelet
    return:
        conv_res: np.ndarray, complex coefficients
    ref:Mik X Cohen, Analyzing Neural Time Series Data: Theory and Practice
    '''

    sig = signal
    fs = fs

    # generating our wavelet
    f = peak_freq
    t = np.arange(-1, 1 + 1/fs, 1/fs)
    s = n/(2*np.pi*f)
    wavelet = np.sqrt(1/(s*np.sqrt(np.pi)))*np.exp(2*np.pi*1j*f*t) * np.exp(-t**2/((2*s)**2))

    # fft params
    n_sig = len(sig)
    n_wavelet = len(wavelet)
    n_conv = n_wavelet + n_sig - 1
    n_conv_pwr2 = 2**(math.ceil(np.log2(np.abs(n_conv))))
    n_half_wavelet = len(wavelet) // 2

    # convolultion
    sig_fft = fft(sig, n_conv_pwr2)
    wavelet_fft = fft(wavelet, n_conv_pwr2)
    conv_result = ifft(sig_fft * wavelet_fft)[:n_conv]#* (np.sqrt(s)/20) # scaling factor = np.squrt(s)/20
    conv_result = conv_result[n_half_wavelet:-n_half_wavelet]   # bc that's what mike used in the text

    return conv_result

def baseline_norm(raw_pwr, bline_index, db=False):
    '''
    takes n by m matrix of raw power and normalizes each frequency to baseline period
    defined by bline_index. if db is False, norm is computed by dividing each sample by the
    mean of the baseline period. is db is True, computes decibel norm instead.
    raw_pwr a list of lists; each sublist is instantaneous power over time (samples)
    for a given frequency in freq range of mwt.
    bline_index denotes the the index at which the baseline period ends and the
    signal begins.

    returns a 2D array of normalized power whose dimensions are n by m - bline_index
    '''
    norm_pwr = []
    for f_sig in raw_pwr:
        sig = f_sig[bline_index:]
        baseline = f_sig[:bline_index]
        b_mean = np.mean(baseline)

        if not db:
            norm_sig = np.array(sig) / b_mean
        elif db:
            norm_sig = 10 * np.log10(sig / b_mean)

        norm_pwr.append(norm_sig)

    return norm_pwr

def main():
    # set up
    # first up, low freq
    label_r1, data_r1 = load_data(args.path_r1)
    label_r2, data_r2 = load_data(args.path_r2)

    r1 = label_r1.split('_')[-1]
    r2 = label_r2.split('_')[-1]
    r1r2label = '_'.join(label_r1.split('_')[:-1] + [f'{r1}-{r2}'])

    freqs = np.arange(2, 20)
    fs = 2000
    b_i = 1000  # 500 ms baseline period right before plot
    t_0 = 600   # 4000 sample epochs sliced at [400:3400] after mwt
                # result is baseline=[400:1400], plot=[1400:3400], t=0 @ 2000
                # or in terms of mwt slice: [0:1000], [1000:3000], t=0 @ 1600; therefore b_i=1000
                # or in terms of plot slice: [-1000:0], [0:2000], t=0 @ 600; therefore t_0=600
    ns = np.linspace(3,6, len(freqs))               # ns for low freq
    #ns = np.linspace(6,12, len(freqs))              # ns for high freq
    n_dict = dict(zip(freqs, ns))

    # loop over regions
    d = [data_r1, data_r2]
    labels = [label_r1, label_r2]

    for data, label in zip(d, labels):
        # compute mwt, extract power, normalize per trial
        norm_trials = []
        for trial_n in data:
            sig = data[trial_n]
            raw_pwr = [abs(compute_mwt(sig, fs, f, n_dict[f]))**2 for f in freqs]
            raw_pwr = [f_pwr[400:3400] for f_pwr in raw_pwr ] # cut off edges we don't want to plot
            norm_pwr = baseline_norm(raw_pwr, b_i, db=False)
            norm_pwr = np.array(norm_pwr)[:,t_0:] # toss into array, keep only data after t=0
            norm_trials.append((trial_n, norm_pwr))

        # separating out time-frequency windows for each trial
        rows = []
        for trial_n, norm_pwr in norm_trials:
            n_half = len(norm_pwr) // 2
            delta_e = np.mean(norm_pwr[2-2:4-2, :n_half]) # _e for early (pre-perception)
            delta_l = np.mean(norm_pwr[2-2:4-2, n_half:]) # _l for late (post-perception)
            theta_e = np.mean(norm_pwr[4-2:10-2, :n_half])
            theta_l = np.mean(norm_pwr[4-2:10-2, n_half:])
            alpha_e = np.mean(norm_pwr[10-2:15-2, :n_half])
            alpha_l = np.mean(norm_pwr[10-2:15-2, n_half:])
            beta_lo_e = np.mean(norm_pwr[15-2:20-2, :n_half])
            beta_lo_l = np.mean(norm_pwr[15-2:20-2, n_half:])

            row = (trial_n, delta_e, delta_l, theta_e, theta_l, alpha_e, alpha_l, beta_lo_e, beta_lo_l)
            rows.append(row)

        if 'PFC' in label:
            r = 'PFC'
        elif 'HPC' in label:
            r = 'HPC'
        df_data = pd.DataFrame(rows, columns = ['trial_n', f'{r}_delta_e', f'{r}_delta_l', f'{r}_theta_e', f'{r}_theta_l', f'{r}_alpha_e', f'{r}_alpha_l', f'{r}_beta_lo_e', f'{r}_beta_lo_l'])
        df_data.to_csv(f'{label}_low_freq.csv')

    ### ======== repeat for high f ======== ###
    freqs = np.arange(20, 100)
    fs = 2000
    b_i = 1000  # 500 ms baseline period right before plot
    t_0 = 600   # 4000 sample epochs sliced at [400:3400] after mwt
                # result is baseline=[400:1400], plot=[1400:3400], t=0 @ 2000
                # or in terms of mwt slice: [0:1000], [1000:3000], t=0 @ 1600; therefore b_i=1000
                # or in terms of plot slice: [-1000:0], [0:2000], t=0 @ 600; therefore t_0=600
    # ns = np.linspace(3,6, len(freqs))               # ns for low freq
    ns = np.linspace(6,12, len(freqs))              # ns for high freq
    n_dict = dict(zip(freqs, ns))

    # loop over regions
    d = [data_r1, data_r2]
    labels = [label_r1, label_r2]

    for data, label in zip(d, labels):
        # compute mwt, extract power, normalize per trial
        norm_trials = []
        for trial_n in data:
            sig = data[trial_n]
            raw_pwr = [abs(compute_mwt(sig, fs, f, n_dict[f]))**2 for f in freqs]
            raw_pwr = [f_pwr[400:3400] for f_pwr in raw_pwr ] # cut off edges we don't want to plot
            norm_pwr = baseline_norm(raw_pwr, b_i, db=False)
            norm_pwr = np.array(norm_pwr)[:,t_0:] # toss into array, keep only data after t=0
            norm_trials.append((trial_n, norm_pwr))

        # separating out time-frequency windows for each trial
        rows = []
        for trial_n, norm_pwr in norm_trials:
            n_half = len(norm_pwr) // 2
            beta_hi_e = np.mean(norm_pwr[20-20:30-20, :n_half])
            beta_hi_l = np.mean(norm_pwr[20-20:30-20, n_half:])
            gamma_lo_e = np.mean(norm_pwr[30-20:60-20, :n_half])
            gamma_lo_l = np.mean(norm_pwr[30-20:60-20, n_half:])
            gamma_mid_e = np.mean(norm_pwr[60-20:80-20, :n_half])
            gamma_mid_l = np.mean(norm_pwr[60-20:80-20, n_half:])
            gamma_hi_e = np.mean(norm_pwr[80-20:100-20, :n_half])
            gamma_hi_l = np.mean(norm_pwr[80-20:100-20, n_half:])

            row = (trial_n, beta_hi_e, beta_hi_l, gamma_lo_e, gamma_lo_l, gamma_mid_e, gamma_mid_l, gamma_hi_e, gamma_hi_l)
            rows.append(row)

        if 'PFC' in label:
            r = 'PFC'
        elif 'HPC' in label:
            r = 'HPC'
        df_data = pd.DataFrame(rows, columns = ['trial_n', f'{r}_beta_hi_e', f'{r}_beta_hi_l', f'{r}_gamma_lo_e', f'{r}_gamma_lo_l', f'{r}_gamma_mid_e', f'{r}_gamma_mid_l', f'{r}_gamma_hi_e', f'{r}_gamma_hi_l'])
        df_data.to_csv(f'{label}_high_freq.csv')

    ### ===== intersite phase clustering ===== ###
    # depending on the version of python, dicts may be unordered
    assert set(data_r1.keys()) == set(data_r2.keys())
    trial_ns = list(data_r1.keys())

    freqs = np.arange(4,40)
    fs = 2000
    b_i = 1000  # 500 ms baseline period right before plot
    t_0 = 600   # 4000 sample epochs sliced at [400:3400] after mwt
                # result is baseline=[400:1400], plot=[1400:3400], t=0 @ 2000
                # or in terms of mwt slice: [0:1000], [1000:3000], t=0 @ 1600; therefore b_i=1000
                # or in terms of plot slice: [-1000:0], [0:2000], t=0 @ 600; therefore t_0=600
    ns = np.linspace(4,9, len(freqs))
    n_dict = dict(zip(freqs, ns))
    sliding_window = 600 # 400 samples, 200 ms sliding window for ispc

    # compute instantaneous phase angles
    pha_r1 = []
    for trial_n in trial_ns:
        sig = data_r1[trial_n]
        pha = [np.angle(compute_mwt(sig, fs, f, n_dict[f])) for f in freqs]
        pha = [f_pha[400+b_i+t_0:3400+sliding_window] for f_pha in pha] # add t_0 here, we only care about data after t=0
        pha_r1.append(pha)

    # compute instantaneous phase angles
    pha_r2 = []
    for trial_n in trial_ns:
        sig = data_r2[trial_n]
        pha = [np.angle(compute_mwt(sig, fs, f, n_dict[f])) for f in freqs]
        pha = [f_pha[400+b_i+t_0:3400+sliding_window] for f_pha in pha] # add t_0 here, we only care about data after t=0
        pha_r2.append(pha)

    delta_pha = np.array(pha_r1) - np.array(pha_r2)

    ispcs = []
    for trial_n, trial in zip(trial_ns, delta_pha):
        trial_ispc = []

        for f in trial:
            f_ispc = []

            for i in range(len(f)-sliding_window):
                window = f[i:i+sliding_window]
                window_ispc = np.abs(np.mean(np.exp(1j * window)))
                f_ispc.append(window_ispc)

            trial_ispc.append(f_ispc)

        ispcs.append((trial_n, np.array(trial_ispc)))

    rows =[]
    for trial_n, trial_ispc in ispcs:
        n_half = len(trial_ispc) // 2
        theta_ispc_e = np.mean(trial_ispc[4-4:10-4, :n_half])
        theta_ispc_l = np.mean(trial_ispc[4-4:10-4, n_half:])
        alpha_ispc_e = np.mean(trial_ispc[10-4:15-4, :n_half])
        alpha_ispc_l = np.mean(trial_ispc[10-4:15-4, n_half:])
        beta_lo_ispc_e = np.mean(trial_ispc[15-4:22-4, :n_half])
        beta_lo_ispc_l = np.mean(trial_ispc[15-4:22-4, n_half:])
        beta_hi_ispc_e = np.mean(trial_ispc[22-4:30-4, :n_half])
        beta_hi_ispc_l = np.mean(trial_ispc[22-4:30-4, n_half:])
        gamma_lo_ispc_e = np.mean(trial_ispc[30-4:40-4, :n_half])
        gamma_lo_ispc_l = np.mean(trial_ispc[30-4:40-4, n_half:])

        row = (trial_n, theta_ispc_e, theta_ispc_l, alpha_ispc_e, alpha_ispc_l, beta_lo_ispc_e, beta_lo_ispc_l, beta_hi_ispc_e, beta_hi_ispc_l, gamma_lo_ispc_e, gamma_lo_ispc_l)
        rows.append(row)

    df_data = pd.DataFrame(rows, columns = ['trial_n', 'theta_ispc_e', 'theta_ispc_l', 'alpha_ispc_e', 'alpha_ispc_l', 'beta_lo_ispc_e', 'beta_lo_ispc_l', 'beta_hi_ispc_e', 'beta_hi_ispc_l', 'gamma_lo_ispc_e', 'gamma_lo_ispc_l'])
    df_data.to_csv(f'{r1r2label}_ispcs.csv')

    ### ====== PAC ====== ###
    # depending on the version of python, dicts may be unordered
    assert set(data_r1.keys()) == set(data_r2.keys())
    trial_ns = list(data_r1.keys())

    freqs_pwr = np.arange(30,101,10)
    freqs_pha = np.arange(6,11,2)
    fs = 2000
    b_i = 1000  # 500 ms baseline period right before plot
    t_0 = 600   # 4000 sample epochs sliced at [400:3400] after mwt
                # result is baseline=[400:1400], plot=[1400:3400], t=0 @ 2000
                # or in terms of mwt slice: [0:1000], [1000:3000], t=0 @ 1600; therefore b_i=1000
                # or in terms of plot slice: [-1000:0], [0:2000], t=0 @ 600; therefore t_0=600
    ns = np.linspace(6,12, len(freqs_pwr))
    n_dict_pwr = dict(zip(freqs_pwr, ns))

    # compute instantaneous power
    pwr_r1 = []
    for trial_n in trial_ns:
        sig = data_r1[trial_n]  # note we use raw power to compute pac, not normalized power
        pwr = [abs(compute_mwt(sig, fs, f, n_dict_pwr[f]))**2 for f in freqs_pwr]
        pwr = [f[400+b_i+t_0:3400] for f in pwr] # add t_0 here, we only care about data after t=0
        pwr_r1.append(pwr)

    ns = np.linspace(3,5, len(freqs_pha))
    n_dict_pha = dict(zip(freqs_pha, ns))

    # compute instantaneous phase angles
    pha_r2 = []
    for trial_n in trial_ns:
        sig = data_r2[trial_n]
        pha = [np.angle(compute_mwt(sig, fs, f, n_dict_pha[f])) for f in freqs_pha]
        pha = [f_pha[400+b_i+t_0:3400] for f_pha in pha] # add t_0 here, we only care about data after t=0
        pha_r2.append(pha)

    rows = []
    for trial_n, pha, pwr in zip(trial_ns, pha_r2, pwr_r1):
        pacs = []
        for f_pwr in pwr:
            for f_pha in pha:
                pac = np.abs(np.mean(f_pwr * np.exp(1j * f_pha)))
                pacs.append(pac)
        row = [trial_n] + pacs
        row = tuple(row)
        rows.append(row)

    pac_cols = []
    for f_pow in freqs_pwr:
        for f_pha in freqs_pha:
            pac_cols.append(f'{f_pow}Hz-{f_pha}Hz pac')
    cols = ['trial_n'] + pac_cols
    data_df = pd.DataFrame(rows, columns = cols)
    data_df.to_csv(f'{r1r2label}_pac.csv')

    sys.exit()

if __name__ == '__main__':
    main()
