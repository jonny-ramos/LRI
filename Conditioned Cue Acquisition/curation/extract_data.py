import glob
import os
import sys
import numpy as np
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser(description='extracts epochs from recordings, checks for artifacts via rms threshold and rejects trials containing artifacts')
parser.add_argument('path', metavar='<path>', help='path of dir containing PFC recording, HPC recording, REF recording, and event timestamps')
args = parser.parse_args()

def get_epochs(sig_file, event_file, ref_file, reference=True):
    '''
    reads in signal, reference and event data from csvs.
    if reference is true, then reference the signal time series by point wise subtraction
    between signal and reference time series.

    round each timestamp to nearest integer index (fs is known to be 2000 for these data)
    and then slices a 2 second epoch around each timestamp (1 sec before, 1 sec after)

    returns a list of lists
    '''
    # read in data
    ch = pd.read_csv(sig_file)
    values = np.array(ch.decimated_values)

    if reference:
        ref = pd.read_csv(ref_file)
        ref_values = np.array(ref.decimated_values)

        values = values - ref_values
        values = list(values)

    # build events dictionary
    df_events = pd.read_csv(event_file)
    d_events = dict(zip(list(df_events.event), list(df_events.timestamp)))

    # convert timestamps in seconds to slices indices
    timestamps = np.around([d_events[key]*2000 for key in d_events]).astype(int)
    timestamp_start = list(timestamps - 2000)
    timestamp_end = [time + (2000*2) for time in timestamp_start]

    # slicing up data
    epochs = []
    for t_start, t_end in zip(timestamp_start, timestamp_end):
        epochs.append(values[t_start:t_end])

    return epochs

def rms(x):
    rms = np.sqrt(np.mean(np.array(x) ** 2))
    return rms

def rms_artifact_ID(trial, stepsize, sampsize, rmsthresh):
    '''
    '''
    rmss = []
    for i in range(int(len(trial)/stepsize)):
        trial_samp = trial[i*stepsize:i*stepsize+sampsize]
        trial_samp_rms = rms(trial_samp)
        rmss.append(trial_samp_rms)

    # without knowing beforehand which sections of data are a clean baseline
    # consider artifacts much shorter than total duration of a given trial, we
    # can guess that the median rms will likely be a reasonable baseline
    rms_baseline = np.median(rmss)

    # True if an artifact greater than threshold was detected
    artifact_filter = rmss > rmsthresh * rms_baseline

    return artifact_filter


def main():
    files = glob.glob(args.path + '/*')
    PFC_path = [f for f in files if 'PFC' in f][0]
    HPC_path = [f for f in files if 'HPC' in f][0]
    REF_path = [f for f in files if 'REF' in f][0]

    # loop twice, once for each cue (coke, saline)
    for i in range(2):
        if i == 0:
            cue = 'coke'
        if i == 1:
            cue = 'saline'

        print('\nloading in data...')
        print(f'{cue} cue ofiles:')
        events = [f for f in files if cue in f][0]

        PFC_ofile = PFC_path.split('/')[-1].replace('.csv', f'_{cue}.npy')
        HPC_ofile = HPC_path.split('/')[-1].replace('.csv', f'_{cue}.npy')

        print(PFC_ofile)
        print(HPC_ofile)

        # read in data, slice out epochs
        PFC_epochs = get_epochs(PFC_path, events, REF_path)
        HPC_epochs = get_epochs(HPC_path, events, REF_path)

        # artifact detection, automatic trial rejection
        step = 1
        samp = 500
        thresh = 1.8
        l = len(PFC_epochs)
        assert len(PFC_epochs) == len(HPC_epochs)

        PFC_artifacts = [rms_artifact_ID(trial, step, samp, thresh) for trial in PFC_epochs]
        HPC_artifacts = [rms_artifact_ID(trial, step, samp, thresh) for trial in HPC_epochs]

        # summing artifact indices across regions
        artifact_sums = [np.array(PFC_a) + np.array(HPC_a) for PFC_a, HPC_a in zip(PFC_artifacts, HPC_artifacts)]

        # artifact removal via boolean indexing
        PFC_cleaned = [np.array(PFC_trial)[~artifact_sum] for PFC_trial, artifact_sum in zip(PFC_epochs, artifact_sums)]
        HPC_cleaned = [np.array(HPC_trial)[~artifact_sum] for HPC_trial, artifact_sum in zip(HPC_epochs, artifact_sums)]

        # trials with no artifacts will still be 4000 samples long
        PFC_cleaned = list(filter(lambda x: len(x) == 4000, PFC_cleaned))
        HPC_cleaned = list(filter(lambda x: len(x) == 4000, HPC_cleaned))

        print(f'trials removed from PFC: {len(PFC_epochs) - len(PFC_cleaned)}')
        print(f'trials removed from HPC: {len(HPC_epochs) - len(HPC_cleaned)}')

        np.save(PFC_ofile, PFC_cleaned)
        np.save(HPC_ofile, HPC_cleaned)

    sys.exit()

if __name__ == '__main__':
    main()
