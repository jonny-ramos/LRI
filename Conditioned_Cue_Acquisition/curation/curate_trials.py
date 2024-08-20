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
        responses = set(['y', 'n', 's', 'quit'])
        #while keep_trial.lower() != 'y' and keep_trial.lower() != 'n' and keep_trial.lower() != 's':

        while keep_trial.lower() not in responses:
            print('That was not a valid response. Please enter Y/N. type help for help')
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

                elif keep_trial.lower() == 'quit':
                    sys.exit()

        elif keep_trial.lower() == 'quit':
            sys.exit()

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

def rms(x):
    return np.sqrt(np.mean(np.array(x) ** 2))

def rat_rms(d_trials):
    rats = ['Ephys5_rat2','Ephys7_rat2','Ephys9_rat2','Ephys10_rat1','Ephys5_rat1','Ephys6_rat1','Ephys9_rat1','Ephys11_rat1']
    keys = list(d_trials.keys())

    rat_rmss = []
    grouped_trials = []
    for rat in rats:
        rat_keys = [k for k in keys if rat in k]
        rat_trials = np.array([d_trials[k] for k in rat_keys])

        if not len(rat_trials) == 0:
            concat = np.concatenate(rat_trials)
            rat_rms = rms(concat)
            rat_rmss.append((rat, rat_rms))

            grouped_trials.append((rat, dict(zip(rat_keys, list(rat_trials)))))

    return dict(rat_rmss), dict(grouped_trials)

def windowed_rms(trial, step=500, n_samples=18000):
    return [rms(trial[i:i+step]) for i in np.arange(0,n_samples,step)]

def rms_filter(PFC_rms, HPC_rms, PFC_trials, HPC_trials, thresh=1.8):
    assert len(PFC_trials) == len(HPC_trials)
    assert PFC_trials.keys() == HPC_trials.keys()

    # PFC filter
    PFC_keys_filt = []
    for key in PFC_trials.keys():
        rmss = windowed_rms(PFC_trials[key])
        if np.all(np.array(rmss) / PFC_rms < thresh):
            PFC_keys_filt.append(key)

    # HPC filter
    HPC_keys_filt = []
    for key in HPC_trials.keys():
        rmss = windowed_rms(HPC_trials[key])
        if np.all(np.array(rmss) / HPC_rms < thresh):
            HPC_keys_filt.append(key)

    # sorted list of set intersection between PFC and HPC filters
    inter = set(PFC_keys_filt).intersection(set(HPC_keys_filt))
    inter = sorted(list(inter), key=lambda x: int(x.split('trial')[-1]))

    # building new dictionaries from filtered key list
    PFC_filt = dict([(k, PFC_trials[k]) for k in inter])
    HPC_filt = dict([(k, HPC_trials[k]) for k in inter])
    print(f'{len(PFC_trials) - len(PFC_filt)} trials removed via rms threshold (rms={thresh})')

    return PFC_filt, HPC_filt

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

    ##### add rms thresholding here again ? maybe.
    d_PFC_rms, d_PFC = rat_rms(PFC)
    d_HPC_rms, d_HPC = rat_rms(HPC)
    assert d_PFC_rms.keys() == d_HPC_rms.keys() == d_PFC.keys() == d_HPC.keys()

    # for each rat, apply rms threshold, build larger dict containing filtered data from all rats
    PFC_filt = {}
    HPC_filt = {}
    for rat in d_PFC_rms.keys():
        PFC_rms, HPC_rms = d_PFC_rms[rat], d_HPC_rms[rat]
        PFC_trials, HPC_trials = d_PFC[rat], d_HPC[rat]

        rat_PFC_filt, rat_HPC_filt = rms_filter(PFC_rms, HPC_rms, PFC_trials, HPC_trials)

        PFC_filt = dict(**PFC_filt, **rat_PFC_filt)
        HPC_filt = dict(**HPC_filt, **rat_HPC_filt)

    print('\nbeginning PFC curation:')
    curated_PFC = curate_trials(PFC_filt)

    print('\nbeginning HPC curation:')
    curated_HPC = curate_trials(HPC_filt)

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
