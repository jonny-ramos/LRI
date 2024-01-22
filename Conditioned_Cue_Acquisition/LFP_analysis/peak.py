import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from lfp import load_data, erp

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data and characterizes N1, P1 ERPs')
parser.add_argument('path', metavar='<path>', help='path of curated data to load')
args = parser.parse_args()

def erp_peak(erp_sig, i_start, i_stop, cut=20, peak='P'):
    '''
    takes array of signals and returns index of highest peak within specified indices
    args:
        sig: np.ndarray(N), N: the number of samples
        i_start: int, index to begin peak search
        i_end: int, index to end peak search
        cut: int, the number of samples to cut, centered on peak index
        peak: str, accepts either 'P' or 'N';
            'P' denotes search for positive peak,
            'N' denotes search for negative peak
    return:
        peak_start: int, index denoting start of peak slice
        peak_stop: int, index denoting end of peaks slice

    '''
    # quickly check type, might be list
    if type(erp_sig) != np.ndarray:
        sig = np.array(erp_sig)

    # slice to search for peak
    erp_search = erp_sig[i_start:i_stop]

    # find peak index, slice around it
    # search for positive peak
    if peak == 'P':
        i_peak = erp_search.argmax() + i_start # add i_start to index from full sig

    # search for negative peak
    elif peak == 'N':
        i_peak = erp_search.argmin() + i_start

    # throw error otherwise
    else:
        raise ValueError('peak must be set to either "P" or "N"')

    peak_start, peak_stop = i_peak - cut // 2, i_peak + cut // 2

    return peak_start, peak_stop

def slice_index(ti, tj, fs, t0):
    '''
    converts timestamps in ms to indices for slicing
    args:
        ti: int, timestamp in ms to begin slicing
        tj: int, timestamp in ms to stop slicing
        fs: int, sampling rate
        t0: int, sample at which time 0 occurs (denoting event)
    '''
    t_start, t_stop = int(ti/1000 * fs + t0), int(tj/1000 * fs + t0)

    return t_start, t_stop

def avg_peak(data, ti, tj):
    '''
    takes data matrix, slices out section, returns 1D arr containing mean of
    each slice
    args:
        data: np.ndarray(N, t); N: the number of samples, t: the number of trials
        ti: int, index denoting beginning of slice
        tj: int, index denoting end of slice
    return:
        peak_means: np.ndarraay(t); t: the number of trials
    '''
    peak_means = []
    for sig in data:
        sig_avg = np.average(sig)
        sig_centered = np.array(sig) - sig_avg
        peak_mean = np.average(sig_centered[ti: tj])
        peak_means.append(peak_mean)

    return np.array(peak_means)


def main():
    label, data = load_data(args.path)
    fs = 2000
    t0 = 2000 # this may not always be true, see data extraction/epoching

    trial_ns = np.array(list(data.keys()))
    data = np.array(list(data.values()))
    n = np.shape(data)[1] # the number of samples

    # compute erp
    erp_sig = erp(data)

    # search for positive peak within first 80-120 ms (P1)
    ti = 80
    tj = 120
    t_start, t_stop = slice_index(ti, tj, fs, t0)
    P1_i, P1_j = erp_peak(erp_sig, t_start, t_stop)
    P1 = avg_peak(data, P1_i, P1_j)

    # search for negative peak within first 150-200 ms (N1)
    ti = 100
    tj = 200
    t_start, t_stop = slice_index(ti, tj, fs, t0)
    N1_i, N1_j = erp_peak(erp_sig, t_start, t_stop, peak='N')
    N1 = avg_peak(data, N1_i, N1_j)

    # search for positive peak within first 200-300 ms (P2?)
    ti = 200
    tj = 300
    t_start, t_stop = slice_index(ti, tj, fs, t0)
    P2_i, P2_j = erp_peak(erp_sig, t_start, t_stop)
    P2 = avg_peak(data, P2_i, P2_j)

    # toss into dataframe, write to csv
    df = pd.DataFrame()
    df['P1'] = P1
    df['N1'] = N1
    df['P2'] = P2

    df.to_csv(f'{label}_peaks.csv')


if __name__ == '__main__':
    main()
