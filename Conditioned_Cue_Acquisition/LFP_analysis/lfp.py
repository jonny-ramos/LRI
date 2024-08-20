import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import argparse
from scipy import stats
from numpy.fft import fft, ifft
from scipy.signal import butter, filtfilt, hilbert

def load_data(file):
    '''
    loads in data file. returns label and dict of curated data.
    '''
    # files = glob.glob('/Users/jonathanramos/Desktop/CCA NEW ANALYSES/curation_2023/data_curated/CUE0/*.npy')
    f = file
    label = '_'.join(f.replace('_curated.npy','').split('/')[-2:])
    data = np.load(f, allow_pickle=True)

    return label, data.item()

def erp(data):
    '''
    takes data matrix(N, L), centers each trial, then computes the erp: sum
    centered trials, divide each signal by number of trials.
    args:
        data(L, N): np.ndarray, L: number of trials, N: number of samples (time
        points)
    return:
        erp(1, N): np.ndarray, N: number of samples (time points)
    '''
    centered_trials = []
    for sig in data:
        sig_avg = np.average(sig)
        sig_centered = np.array(sig) - sig_avg
        centered_trials.append(sig_centered)

    return np.array(centered_trials).mean(axis=0)

# morlet wavelet convolution
def compute_mwt(signal, fs, peak_freq, n):
    '''
    Takes a timeseries and computes morlet wavelet convolution.
    args:
        signal(N): list, N: number of samples (time points)
        fs: int, sampling rate
        peak_freq: int, peak frequency of generated wavelet
        n: int, number of cycles in generated wavelet
    return:
        conv_result: np.ndarray, copmlex coefficients resulting from convolution
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
    conv_result = conv_result[n_half_wavelet:-n_half_wavelet]

    return conv_result

# linear normalization
def baseline_norm(raw_pwr, bline_index, db=False):
    '''
    takes n by m matrix of raw power and normalizes each frequency to baseline period
    defined by bline_index. if db is False, norm is computed by dividing each sample by the
    mean of the baseline period. is db is True, computes decibel norm instead.
    raw_pwr a list of lists; each sublist is instantaneous power over time (samples)
    for a given frequency in freq range of mwt.
    bline_index denotes the the index at which the baseline period ends and the
    signal begins.

    args:
        raw_pwr(t,N): np.ndarray, t: number of trials, N: number of samples
        bline_index: int, denotes index where trial begins and baseline ends
    return:
        norm_pwr: np.npdarray(t,N-bline_index), note section of data considered
            as baseline is not returned
    '''
    norm_pwr = []
    for f_sig in raw_pwr:

        sig = f_sig[bline_index:]
        baseline = f_sig[:bline_index]
        b_mean = np.average(baseline)

        if not db:
            norm_sig = np.array(sig) / b_mean
        elif db:
            norm_sig = 10 * np.log10(sig / b_mean)

        norm_pwr.append(norm_sig)

    return norm_pwr

# intertrial phase clustering
def itpc(angles):
    '''
    small fn to quickly compute ITPC in the shape of our data (list of list of lists)
    '''
    angles_T = np.transpose(angles)

    ITPCs = []
    for sig in angles_T:

        ITPC_byfreq = []
        for freq in sig:
            # compute ITPC
            ITPC_byfreq.append(np.abs(np.mean(np.exp(1j * freq))))

        ITPCs.append(ITPC_byfreq)

    ITPCs = np.transpose(ITPCs)
    return ITPCs

# intersite phase clustering
def ispc(R1_angles, R2_angles):
    R1_angles_T = np.transpose(R1_angles)
    R2_angles_T = np.transpose(R2_angles)

    # finding our angle differences
    R1_R2_angles_diff_T = R1_angles_T - R2_angles_T

    R1_R2_ISPCs = []

    for sig in R1_R2_angles_diff_T:
        ISPC_byfreq = []

        for freq in sig:
            # compute ISPC
            ISPC_byfreq.append(np.abs(np.mean(np.exp(1j * freq))))

        R1_R2_ISPCs.append(ISPC_byfreq)

    R1_R2_ISPCs = np.transpose(R1_R2_ISPCs)

    return R1_R2_ISPCs


def pli(R1_angles, R2_angles):
    '''
    R1_angles, R2_angles are timeseries of instantaneous phase angles from respective
    regions R1 and R2 for a list of trials. Both are expected to be a 2D array organized
    first by f then by sample.
    '''
    R1_angles_T = np.transpose(R1_angles)
    R2_angles_T = np.transpose(R2_angles)

    # per sample angle differences
    R1_R2_angles_diff_T = R1_angles_T - R2_angles_T

    R1_R2_PLIs =[]
    for sig in R1_R2_angles_diff_T:

        PLI_byfreq = []
        for freq in sig:
            # compute PLI
            PLI_byfreq.append(np.abs(np.mean(np.sign(np.imag(np.exp(1j * freq))))))

        R1_R2_PLIs.append(PLI_byfreq)

    R1_R2_PLIs = np.transpose(R1_R2_PLIs)

    return R1_R2_PLIs

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)

    return y

def filter_hilbert(sig, lc, hc, fs, order):
    filt = butter_bandpass_filter(sig, lc, hc, fs, order)
    asig = hilbert(filt)

    return asig

def bin_pha(epoch):
    epoch_bin_bool = []
    for i in np.arange(-15,15):
        bin_low = (np.pi*i)/15
        bin_high = (np.pi*(i+1))/15
        bin1 = (bin_low < epoch) & (epoch < bin_high)

        epoch_bin_bool.append(bin1.tolist())

    # getting indices where True
    epoch_bin_indices = [np.argwhere(binned_epoch).tolist() for binned_epoch in epoch_bin_bool]

    return epoch_bin_bool


def modulation_index(binned_pwr, pha_bins):
    n = len(pha_bins)

    # construct p(j): normalize by dividing each bin value by the sum over the bins
    p = np.array(binned_pwr) / np.sum(binned_pwr)

    # Shannon entropy H(p)
    h = (-1)*np.sum([np.log(p_j)*p_j for p_j in p])

    # Kullback-Leibler distance KL
    kl = np.log(len(pha_bins)) - h

    # Tort et al 2008 Modulation Index
    return kl / np.log(len(pha_bins))
