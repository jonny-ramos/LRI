import glob
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='executes specified analyses on epoched, cleaned, curated ephys recording data.')
parser.add_argument('-sp', '--spectrogram', action='store_true', help='if true, compute and plot spectrogram')
parser.add_argument('-it', '--itpc', action='store_true', help='if true, compute and plot itpcs')
parser.add_argument('-is', '--ispc', action='store_true', help='if true, compute and plot ispcs')
parser.add_argument('-pl', '--pli', action='store_true', help='if true, compute and plot plis')
parser.add_argument('-er', '--erp', action='store_true', help='if true, compute and plot erps')
parser.add_argument('-pd', '--pwrphadist', action='store_true', help='if true, compute and plot distribution of average power over binned phase')
parser.add_argument('-wn', '--window', action='store_true', help='if true, compute mean pwr and coupling metrics over time freq windows, write to csv')
parser.add_argument('-xc', '--lagxcorr', action='store_true', help='if true, compute pwr envelope for each region, plot lag cross correlations')
parser.add_argument('-xct', '--lagxcorr_trials', action='store_true', help='if true, compute pwr envelope for each region, plot lag cross correlations by trial')
parser.add_argument('-pac', '--pacz', action='store_true', help='if true, compute z-scored pac over designated time-frequency window')
args = parser.parse_args()

# paths
f = '/Users/jonathanramos/Desktop/CCA_2023/curation/data_curated/'
cue0 = f + '/CUE0_lte2400/*.npy'
cue1 = f + '/CUE1_lte2400/*.npy'
T1 = f + '/T1/*.npy'
T2 = f + '/T2/*.npy'
T7 = f + '/T7/*.npy'
T8 = f + '/T8/*.npy'
cue0_long = f + '/CUE0_lte2400_long/*.npy'
cue1_long = f + '/CUE1_lte2400_long/*.npy'
T1_long = f + '/T1_long/*.npy'
T2_long = f + '/T2_long/*.npy'
T7_long = f + '/T7_long/*.npy'
T8_long = f + '/T8_long/*.npy'

for path in [cue0, cue1]:
    if args.spectrogram:
    # CUE0
        files = glob.glob(path)
        if not len(files) == 0:
            for f in files:
                os.system(f'python3 spct.py {f} 2 30')
                os.system(f'python3 spct.py {f} 2 101 -nlow')


    if args.itpc:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:
            for f in files:
                os.system(f'python3 itpc.py {f} 4 40 -nlow')

    if args.ispc:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 ispc.py {PFC} {HPC} 4 40 -nlow')

    if args.pli:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 pli.py {PFC} {HPC} 4 40 -nlow')
    if args.erp:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:
            for f in files:
                os.system(f'python3 erp.py {f}')

    if args.pwrphadist:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 pwrphadist.py {PFC} {HPC} 4,12 80,100')

    if args.window:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 window_trials.py {PFC} {HPC}')

    if args.lagxcorr:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 lagxcorr.py {PFC} {HPC} 4,12')

    if args.pacz:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 pacz.py {PFC} {HPC} 4,12 20,100')

    if args.lagxcorr_trials:
        # CUE0
        files = glob.glob(path)
        if not len(files) == 0:

            # parse out set of partial match strings which groups data across regions
            partial_match = set(['_'.join(os.path.split(f)[1].split('_')[:2]) for f in files])
            for match in partial_match:
                HPC, PFC = tuple(sorted(list(filter(lambda x: match in x, files))))
                os.system(f'python3 lagxcorr_bytrial.py {PFC} {HPC} 4,12')
