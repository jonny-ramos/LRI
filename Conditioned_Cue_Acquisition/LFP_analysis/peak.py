import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from lfp import load_data

parser = argparse.ArgumentParser(description='loads in dict of trialized recording data and characterizes N1, P1 ERPs')
parser.add_argument('path', metavar='<path>', help='path of curated data to load')
args = parser.parse_args()

def get_peak(sig, i_start, i_stop, cut=20, peak='P'):
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
        avg_peak_slice: float, the average value of peak slice

    '''
    # quickly check type, might be list
    if type(sig) != np.ndarray:
        sig = np.array(sig)

    # mean center
    avg = np.average(sig)
    sig_centered = sig - avg
    sig_search = sig_centered[i_start:i_stop]

    # find peak index, slice around it
    # search for positive peak
    if peak == 'P':
        i_peak = sig_search.argmax()

    # search for negative peak
    elif peak == 'N':
        i_peak = sig_search.argmin()

    # throw error otherwise
    else:
        raise ValueError('peak must be set to either "P" or "N"')

    peak_slice = sig_search[i_peak - cut // 2: i_peak + cut // 2]

    # if peak occurred at either end of search space, return avg of whole search space
    if len(peak_slice) == 0:
        peak_slice = sig_search

    return np.average(peak_slice)

def slice_index(ti, tj, fs, t0):
    '''
    converts timestamps in ms to indices for slicing
    arg:
        ti: int, timestamp in ms to begin slicing
        tj: int, timestamp in ms to stop slicing
        fs: int, sampling rate
        t0: int, sample at which time 0 occurs (denoting event)
    '''
    t_start, t_stop = int(ti/1000 * fs + t0), int(tj/1000 * fs + t0)

    return t_start, t_stop


def main():
    label, data = load_data(args.path)
    fs = 2000

    trial_ns = np.array(list(data.keys()))
    data = np.array(list(data.values()))
    n = np.shape(data)[1] # the number of samples

    t0 = 2000 # this may not always be true, see data extraction/epoching

    # search for positive peak within first 80-120 ms (P1)
    ti = 50
    tj = 150
    t_start, t_stop = slice_index(ti, tj, fs, t0)
    P1 = np.array([get_peak(trial, t_start, t_stop) for trial in data])

    # search for negative peak within first 150-200 ms (N1)
    ti = 100
    tj = 200
    t_start, t_stop = slice_index(ti, tj, fs, t0)
    N1 = np.array([get_peak(trial, t_start, t_stop, peak='N') for trial in data])

    # search for positive peak within first 200-300 ms (P2?)
    ti = 200
    tj = 300
    t_start, t_stop = slice_index(ti, tj, fs, t0)
    P2 = np.array([get_peak(trial, t_start, t_stop) for trial in data])

    # toss into dataframe, write to csv
    df = pd.DataFrame()
    df['P1'] = P1
    df['N1'] = N1
    df['P2'] = P2

    df.to_csv(f'{label}_peaks.csv')


if __name__ == '__main__':
    main()