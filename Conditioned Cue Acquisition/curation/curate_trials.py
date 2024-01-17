'''
small script to visually curate trials. data from two regions are loaded and
are curated individually first. returns only trials that were visually curated
in both regions. this means that both data from both regions for a given
day/treatment/cue combination will have the same shape.
'''

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import glob
import sys

parser = argparse.ArgumentParser(description = 'visually curate trialized ephys data one trial at a time')
parser.add_argument('PFCpath', metavar='<path>', help='path to dict of PFC data to curate.')
parser.add_argument('HPCpath', metavar='<path>', help='path to dict of HPC data to curate.')
args = parser.parse_args()

def curate_trials(d_trials):
    sys.stdout = sys.__stdout__

    keep_trials = {}
    for key, trial in d_trials.items():
        plt.figure(figsize=(15,3))
        plt.plot(trial)
        plt.title(key)
        plt.pause(.001)

        keep_trial = input('Keep trial? Y/N: ')
        while keep_trial.lower() != 'y' and keep_trial.lower() != 'n' and keep_trial.lower() != 's':
            print('That was not a valid response. Please enter Y/N')
            keep_trial = input('Keep trial? Y/N: ')

        # keep trial as val if user inputs yes
        if keep_trial.lower() == 'y':
            keep_trials.update({key: trial})

        # val is None if user inputs no
        elif keep_trial.lower() == 'n':
            keep_trials.update({key: None})

        elif keep_trial.lower() == 's':
            print('saving plot...')
            plt.savefig(f'{key}.png', dpi=600)

            keep_trial = input('Keep trial? Y/N: ')
            while keep_trial.lower() != 'y' and keep_trial.lower() != 'n':
                if keep_trial.lower() == 's':
                    print('plot already saved!')
                    keep_trial = input('Keep trial? Y/N: ')
                else:
                    print('That was not a valid response. Please enter Y/N')
                    keep_trial = input('Keep trial? Y/N: ')

                # keep trial as val if user inputs yes
                if keep_trial.lower() == 'y':
                    keep_trials.update({key: trial})

                # val is None if user inputs no
                elif keep_trial.lower() == 'n':
                    keep_trials.update({key: None})


        plt.close()

    return keep_trials

# small fn to correctly maintain keys whose values were not None across both
# the curr dict and the ref dict (val also not None in dict for opposite region)
def dictfilt(curr_d, ref_d):
    assert set(curr_d.keys()) == set(ref_d.keys())

    filt_d = {}
    for key in curr_d:
        if curr_d[key] is not None and ref_d[key] is not None:
            filt_d.update({key: curr_d[key]})

    return filt_d

def main():
    '''
    filteres two regions at a time, then merges keys where val != False
    resulting dictionaries contain only key/val pairs where both val is not False
    in both data sets. Otherwise trial is removed from final curated set. writes
    dicts to disk.
    '''
    print(f'\nloading data from: \n{args.PFCpath} \n{args.HPCpath}')
    PFC = np.load(args.PFCpath, allow_pickle=True).item()
    HPC = np.load(args.HPCpath, allow_pickle=True).item()

    PFC_ofile = args.PFCpath.replace('.npy', '_curated.npy')
    HPC_ofile = args.HPCpath.replace('.npy', '_curated.npy')

    assert len(PFC) == len(HPC)
    assert set(PFC.keys()) == set(HPC.keys())

    print('\nbeginning PFC curation:')
    curated_PFC = curate_trials(PFC)

    print('\nbeginning HPC curation:')
    curated_HPC = curate_trials(HPC)

    print('\nnow merging curated curated trials across regions...')
    filtered_PFC = dictfilt(curated_PFC, curated_HPC)
    filtered_HPC = dictfilt(curated_HPC, curated_PFC)

    print('\nwriting to disk...')
    np.save(PFC_ofile, filtered_PFC)
    np.save(HPC_ofile, filtered_HPC)
    print('\ncuration complete!\n')

    sys.exit()

if __name__ == '__main__':
    main()
