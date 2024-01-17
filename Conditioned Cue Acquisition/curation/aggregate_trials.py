import numpy as np
import argparse
import itertools
import glob
import sys

parser = argparse.ArgumentParser(description='aggregates extracted epochs for curation')
parser.add_argument('path', metavar='<path>', help='path of dir containing extracted epochs in binary format')
args = parser.parse_args()

def partial_match(s, target):
    split = s.split('_')
    for string in split:
        if target in string:
            return string

def get_treatment(fname):
    '''
    small fn to map fname to rat_n to treatment ("ChABC" or "Vehicle")
    '''
    f = fname.split('/')[-1]
    group = partial_match(f, 'Ephys')
    rat = partial_match(f, 'rat')
    rat_n = '_'.join([group, rat])

    ChABC = {'Ephys5_rat2','Ephys7_rat2','Ephys9_rat2','Ephys10_rat1'}

    Vehicle = {'Ephys5_rat1','Ephys6_rat1','Ephys9_rat1','Ephys11_rat1'}

    if rat_n in ChABC:
        return 'ChABC'

    elif rat_n in Vehicle:
        return 'Vehicle'

    else:
        print('check your rat_ns')
        sys.exit(1)

def flatten(kv):
    '''
    for a key/val pair where val is a 2D array, flattens specified key/val pair
    into a new list of key/val pairs where new vals are subarrays of specified
    input val; keys are preserved for all new key/val pairs.
    '''
    key, value = kv
    return [(key, subarray) for subarray in value]

def main():
    files = glob.glob(args.path + '/*.npy')
    treatment = {'ChABC', 'Vehicle'}
    cue = {'coke', 'saline'}
    region = {'PFC', 'HPC'}

    for comb in itertools.product(treatment, cue, region):
        t, c, r = comb
        files_i = [f for f in files if get_treatment(f) == t and c in f and r in f]
        arrs = [(f.split('/')[-1], np.load(f, allow_pickle=True)) for f in files_i]

        ### kind of a makeshift implementation of a flat map in pytho
        flattened_arr = []
        for arr in arrs:
            flat_arr = flatten(arr)
            flat_arr = [('_'.join([partial_match(kv[0], 'Ephys'), partial_match(kv[0], 'rat'), f'trial{i}']), kv[1]) for i, kv in enumerate(flat_arr)]
            flattened_arr += flat_arr

        d_trials = dict(flattened_arr)

        np.save('_'.join(list(comb))+'.npy', d_trials)

    sys.exit()

if __name__ == '__main__':
    main()
