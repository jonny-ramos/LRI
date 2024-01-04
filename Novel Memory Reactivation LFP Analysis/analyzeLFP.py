'''
VR5 Ephys LFP analysis
12/6/22, edited 1/17/23: rewritten in the form of script, streamlined workflow of accessing L/R data
this script will expect to receive data as a binary npy file that has already
been unshelved, bandpass filtered and decimated
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging
import argparse
import datetime
import random
import math
from numpy.fft import fft, ifft
from numpy.linalg import inv
from numpy.linalg import cholesky
from scipy import stats
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import hilbert
from scipy import signal
from scipy import stats

from LFP import *

def main():
    # I expect paths to be of the form '/Volumes/T7/VR5Ephys/VR5/Vehicle/unshelved_dicts/'
    # tail of directory should be 'unshelved_dicts/'

    #####  this section can be reworked using argparser  ######
    parser = argparse.ArgumentParser(description = 'Clean and curate ephys recording data and/or perform various LFP analyses.')
    parser.add_argument('command', metavar='<command>', choices=['curate', 'analyze'], help='command to excecute, choose from curate or analyze.')
    parser.add_argument('path', metavar='<path>', help='path to directory containing data to clean or analyze. i.e. /Volumes/T7/VR5Ephys/VR5/Vehicle/unshelved_dicts/')
    parser.add_argument('s_pre', type=float, metavar='<s_pre>', help='for each epoch, the number of seconds to slice before the timestamp')
    parser.add_argument('s_post', type=float, metavar='<s_post>', help='for each epoch, the number of seconds to slice after the timestamp')
    parser.add_argument('regions', metavar='<regions>', choices=['single', 'pfc-hpc', 'pfc-bla', 'bla-hpc', 'pfc-pfc', 'hpc-hpc', 'within-pfc', 'within-hpc', 'pfc-pfc_r'], help='the regions to analyze', type=str.lower)
    parser.add_argument('global_labels', metavar='<global_labels>', help='denotes treatment and recording session for all ns, i.e. Vehicle_VR5')
    parser.add_argument('-e', '--exclude', action='store', help='denotes rats to exclude for all analyses. Please use full rat_n names seprated by single commmas, i.e. VR5Ephys5_rat1,VR5Ephys5_rat2')
    parser.add_argument('-q', '--quantify', action='store', help='denotes whether or not to quantify contourf plots')
    parser.add_argument('-a', '--analysis', action='store', help='denotes analysis to perform. If not, then perform all analyses')
    parser.add_argument('-hm', '--hemisphere', action='store', help='denotes which hemisphere to conduct multiregional analysis; expects "l", or "r". if not, then perform analyses on both hemispheres.')
    args = parser.parse_args()




    if args.command == 'curate':
        # note on usage: only clean single region data then run analyses on cleaned/curated data.
        # every region pairing can be generated from the single region set, which we can do after cleaning/curation
        # this also minimizes the time spent on trial rejection (if we just do it once and then group data for multiregional analyses.)

        try:
            files = [args.path + d for d in os.listdir(args.path) if '._' not in d] # not sure why there are some hidden files here
            file_dict  = []
            for f in files:
                d = np.load(f, allow_pickle=True).item()

                # with open('check_dict.txt', 'a+') as sys.stdout:
                #     print('============================\n')
                #     print(f.split('/')[-1][:-4], '\n')
                #     print(d, '\n')

                file_dict.append((f.split('/')[-1][:-4], d))        # key is name of tail w/o extension

            # important vars
            DATA = dict(file_dict)
            global RAT_ID
            RAT_ID = np.unique(['_'.join([key.split('_')[0],key.split('_')[1]]) for key in DATA.keys()])

        except FileNotFoundError as e:
            print('Unable to load in data. File not found.')
            sys.exit(1)

        if not args.exclude is None:
            exclude = args.exclude.split(',')
            RAT_ID = [rat for rat in RAT_ID if not rat in exclude]



        #### Here i will include an optional CL arg to denote which rats (if any) are to be
        #### removed from the set before beginning any analyses

        # if len(sys.argv == some int (make sure this is the last var to expect))

        if 'bla' in args.regions.lower() and 'VR5Ephys7_rat2' in RAT_ID:
        # load in VR5Ephys7_rat2 separately for any multiregional analyses involving BLA
        # bc this rat had BLA only in R side.
            RAT_ID = [rat for rat in RAT_ID if rat != 'VR5Ephys7_rat2']

            VR5E7R2_PFC, VR5E7R2_BLA, VR5E7R2_HPC, VR5E7R2_REF = norm_timeseries(DATA, 'VR5Ephys7_rat2', ref_side='L', HPC_ch=True, BLA_ch=True)

            PFC_VR5E7R2 = {'VR5Ephys7_rat2' : VR5E7R2_PFC}
            BLA_VR5E7R2 = {'VR5Ephys7_rat2' : VR5E7R2_BLA}
            HPC_VR5E7R2 = {'VR5Ephys7_rat2' : VR5E7R2_HPC}
            REF_VR5E7R2 = {'VR5Ephys7_rat2' : VR5E7R2_REF}      # we don't actually need this data, just check that the referenced timeseries fin REF is 0

            VR5E7R2_PFC_Rside_rewarded, VR5E7R2_PFC_Lside_rewarded = slice_by_side(PFC_VR5E7R2, args.s_pre, args.s_post, RAT_ID, rewarded=True)
            VR5E7R2_PFC_Rside_unrewarded, VR5E7R2_PFC_Lside_unrewarded = slice_by_side(PFC_VR5E7R2, args.s_pre, args.s_post, RAT_ID, rewarded=False)

            VR5E7R2_BLA_Rside_rewarded, VR5E7R2_BLA_Lside_rewarded = slice_by_side(BLA_VR5E7R2, args.s_pre, args.s_post, RAT_ID, rewarded=True)
            VR5E7R2_BLA_Rside_unrewarded, VR5E7R2_BLA_Lside_unrewarded = slice_by_side(BLA_VR5E7R2, args.s_pre, args.s_post, RAT_ID, rewarded=False)

            VR5E7R2_HPC_Rside_rewarded, VR5E7R2_HPC_Lside_rewarded = slice_by_side(HPC_VR5E7R2, args.s_pre, args.s_post, RAT_ID, rewarded=True)
            VR5E7R2_HPC_Rside_unrewarded, VR5E7R2_HPC_Lside_unrewarded = slice_by_side(HPC_VR5E7R2, args.s_pre, args.s_post, RAT_ID, rewarded=False)


        PFC = {}
        BLA = {}
        HPC = {}
        REF = {}

        for rat in RAT_ID:
            if rat == 'VR5Ephys3_rat2':
                PFC_rat, BLA_rat, REF_rat = norm_timeseries(DATA, rat, ref_side='L', HPC_ch=False, BLA_ch=True)

                PFC[rat] = PFC_rat
                BLA[rat] = BLA_rat
                REF[rat] = REF_rat

            else:
                PFC_rat, BLA_rat, HPC_rat, REF_rat = norm_timeseries(DATA, rat, ref_side='L', HPC_ch=True, BLA_ch=True)

                PFC[rat] = PFC_rat
                BLA[rat] = BLA_rat
                HPC[rat] = HPC_rat
                REF[rat] = REF_rat

        # here's our raw data organized by region/site.
        PFC_Rside_rewarded, PFC_Lside_rewarded = slice_by_side(PFC, args.s_pre, args.s_post, RAT_ID, rewarded=True)
        PFC_Rside_unrewarded, PFC_Lside_unrewarded = slice_by_side(PFC, args.s_pre, args.s_post, RAT_ID, rewarded=False)

        BLA_Rside_rewarded, BLA_Lside_rewarded = slice_by_side(BLA, args.s_pre, args.s_post, RAT_ID, rewarded=True)
        BLA_Rside_unrewarded, BLA_Lside_unrewarded = slice_by_side(BLA, args.s_pre, args.s_post, RAT_ID, rewarded=False)

        HPC_Rside_rewarded, HPC_Lside_rewarded = slice_by_side(HPC, args.s_pre, args.s_post, RAT_ID, rewarded=True)
        HPC_Rside_unrewarded, HPC_Lside_unrewarded = slice_by_side(HPC, args.s_pre, args.s_post, RAT_ID, rewarded=False)

        if 'bla' in args.regions.lower():
        # load in VR5Ephys7_rat2 separately for any multiregional analyses involving BLA.
        # now adding VR5Ephys7_rat2 to our larger dict of data, but adding R side data from
        # VR5Ephys7_rat2 to Lside dicts only bc, this rat had BLA only in R side.
        # and all the other rats had BLA only on L side; so we group the sides into a single
        # dictionary and assume that the L and R sides do the same thing.

        # just going to group on Lside, and then only consider "Lside" for multiregional analyses
        # (but ofc it's a mix of L and R sides, just needed to ensure that for any side taken,
        # that the same side was taken for all other args.regions)

            PFC_Lside_rewarded['VR5Ephys7_rat2'] = VR5E7R2_PFC_Rside_rewarded['VR5Ephys7_rat2']
            PFC_Lside_unrewarded['VR5Ephys7_rat2'] = VR5E7R2_PFC_Rside_unrewarded['VR5Ephys7_rat2']

            BLA_Lside_rewarded['VR5Ephys7_rat2'] = VR5E7R2_BLA_Rside_rewarded['VR5Ephys7_rat2']
            BLA_Lside_unrewarded['VR5Ephys7_rat2'] = VR5E7R2_BLA_Rside_unrewarded['VR5Ephys7_rat2']

            HPC_Lside_rewarded['VR5Ephys7_rat2'] = VR5E7R2_HPC_Rside_rewarded['VR5Ephys7_rat2']
            HPC_Lside_unrewarded['VR5Ephys7_rat2'] = VR5E7R2_HPC_Rside_unrewarded['VR5Ephys7_rat2']

            with open('BLA_exception_append.txt', 'w') as sys.stdout:
                print('other rats: BLA Lside')
                print(BLA_Lside_rewarded)
                print('\n=====================\n')
                print('VR5E7R2: BLA Rside')
                print(VR5E7R2_BLA_Rside_rewarded)

        # ok let's format a semi nice-looking readable summary csv
        df_summary = pd.DataFrame()

        r = [
        'PFC_R_rw_cleaned', 'PFC_L_rw_cleaned', 'PFC_R_unrw_cleaned', 'PFC_L_unrw_cleaned',
        'BLA_R_rw_cleaned', 'BLA_L_rw_cleaned', 'BLA_R_unrw_cleaned', 'BLA_L_unrw_cleaned',
        'HPC_R_rw_cleaned', 'HPC_L_rw_cleaned', 'HPC_R_unrw_cleaned', 'HPC_L_unrw_cleaned'
        ]
        df_summary['region'] = r

        for key in list(PFC_Rside_rewarded.keys()):
            ns = []

            ns.append(len(PFC_Rside_rewarded[key])) if key in PFC_Rside_rewarded.keys() else ns.append(0)
            ns.append(len(PFC_Lside_rewarded[key])) if key in PFC_Lside_rewarded.keys() else ns.append(0)
            ns.append(len(PFC_Rside_unrewarded[key])) if key in PFC_Rside_unrewarded.keys() else ns.append(0)
            ns.append(len(PFC_Lside_unrewarded[key])) if key in PFC_Lside_unrewarded.keys() else ns.append(0)

            ns.append(len(BLA_Rside_rewarded[key])) if key in BLA_Rside_rewarded.keys() else ns.append(0)
            ns.append(len(BLA_Lside_rewarded[key])) if key in BLA_Lside_rewarded.keys() else ns.append(0)
            ns.append(len(BLA_Rside_unrewarded[key])) if key in BLA_Rside_unrewarded.keys() else ns.append(0)
            ns.append(len(BLA_Lside_unrewarded[key])) if key in BLA_Lside_unrewarded.keys() else ns.append(0)

            ns.append(len(HPC_Rside_rewarded[key])) if key in HPC_Rside_rewarded.keys() else ns.append(0)
            ns.append(len(HPC_Lside_rewarded[key])) if key in HPC_Lside_rewarded.keys() else ns.append(0)
            ns.append(len(HPC_Rside_unrewarded[key])) if key in HPC_Rside_unrewarded.keys() else ns.append(0)
            ns.append(len(HPC_Lside_unrewarded[key])) if key in HPC_Lside_unrewarded.keys() else ns.append(0)

            df_summary[key] = ns

        df_summary['sum'] = np.zeros(len(df_summary))   # adding a last column which contains sum trial_ns
        for col in list(df_summary.columns)[1:-1]:
            df_summary['sum'] += df_summary[col]
        df_summary['sum'] = df_summary['sum'].astype(int)

        df_summary.to_csv('_'.join([args.global_labels, args.regions.upper(), 'TOTAL','trial_ns'])+'.csv')


        # THE WAY WE FILTER VIA TRIAL REJECTION DEPENDS ON WHICH TYPE OF MULTIREGIONAL ANALYSIS WE WANT TO DO
        # to analyze data across multiple args.regions, we must ensure that trials from both args.regions
        # were clean for each trial; i.e. if for some trial, there's an artifact in one region but not the other,
        # we want to reject that trial.
        # of course this is not an issue for single region analyses; we take everything we can get for those

        # paramters in case we want to try changing these later.
        STEPSIZE = 1
        SAMPSIZE = 500
        RMSTHRESH = 2.6

        # ok previously I had been storing trial data as a list of lists, with no identifying information attached to each trial
        # now I've switched to storing trial data in dictionaries, accessing a list of trial per animal (per region) as the following:
        # ex. PFC[rat] will give me a list of trials (which are lists)
        #
        # all of my previous functions expect a list of trials. so now I prbably have to do something like
        # for rat in PFC:
            # clean data .... PFC[rat]

        # but rather than storing artifact IDs in lists, I'll have to put them in the same structure as the data.

        PFC_R_rw_artifacts = get_artifacts(PFC_Rside_rewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        PFC_L_rw_artifacts = get_artifacts(PFC_Lside_rewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        PFC_R_unrw_artifacts = get_artifacts(PFC_Rside_unrewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        PFC_L_unrw_artifacts = get_artifacts(PFC_Lside_unrewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)

        # FOR BLA ONLY CONSIDER LSIDE ONLY
        BLA_R_rw_artifacts = get_artifacts(BLA_Rside_rewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        BLA_L_rw_artifacts = get_artifacts(BLA_Lside_rewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        BLA_R_unrw_artifacts = get_artifacts(BLA_Rside_unrewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        BLA_L_unrw_artifacts = get_artifacts(BLA_Lside_unrewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)

        # For any Vehicle HPC, exclude VR5Ephys3_rat2 (this animal did not have any HPC)
        HPC_R_rw_artifacts = get_artifacts(HPC_Rside_rewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        HPC_L_rw_artifacts = get_artifacts(HPC_Lside_rewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        HPC_R_unrw_artifacts = get_artifacts(HPC_Rside_unrewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)
        HPC_L_unrw_artifacts = get_artifacts(HPC_Lside_unrewarded, STEPSIZE, SAMPSIZE, RMSTHRESH)

        # with open('artifact_test.txt', 'w') as sys.stdout:
        #     print(len(PFC_L_rw_artifacts))
        #     print(len(PFC_Lside_rewarded))
        #     print(PFC_L_rw_artifacts)

        ##### ok great !! now comes another tricky bit: now we must keep track of which trials get removed.
        # we can do this by taking the index from enumerate(trials) and setting them as keys in another dictionary
        # this will result (in both the trials dict and the artifacts dict) in a nested dict
        # where the keys are rat_ns and the value is a dictionary of trials
        # where the keys are the index of the trial and the value is the trial (which is a list)
        # something like: PFC[rat][0] should yield the first trial in the PFC for that rat.
        # ...
        # well actually now that I think about it I can probably do this after summing the artifact booleans
        # no need to overcomplicate things just yet...

        # these artifact 'filters' will be combined in different ways depending on the analysis.
        # let's layout all the possible multiregional possibilities we're interested in:
        # PFC-HPC       PFC-BLA     HPC-BLA?    PFC_L-PFC_R
        # as well as all the single args.regions:    PFC     BLA     HPC

        # I think it would be best to take a command line argument to differentiate each of these use cases.

        if args.regions.lower() == 'single':
            # no changes to make for single region analyses (we can keep the max n of trials from each region)
            PFC_R_rw_afilt = PFC_R_rw_artifacts
            PFC_L_rw_afilt = PFC_L_rw_artifacts
            PFC_R_unrw_afilt = PFC_R_unrw_artifacts
            PFC_L_unrw_afilt = PFC_L_unrw_artifacts

            BLA_R_rw_afilt = BLA_R_rw_artifacts
            BLA_L_rw_afilt = BLA_L_rw_artifacts
            BLA_R_unrw_afilt = BLA_R_unrw_artifacts
            BLA_L_unrw_afilt = BLA_L_unrw_artifacts

            HPC_R_rw_afilt = HPC_R_rw_artifacts
            HPC_L_rw_afilt = HPC_L_rw_artifacts
            HPC_R_unrw_afilt = HPC_R_unrw_artifacts
            HPC_L_unrw_afilt = HPC_L_unrw_artifacts

        elif args.regions.lower() == 'pfc-hpc':
            R_rw_afilt = sum_artifact_dicts(PFC_R_rw_artifacts, HPC_R_rw_artifacts)
            L_rw_afilt = sum_artifact_dicts(PFC_L_rw_artifacts, HPC_L_rw_artifacts)
            R_unrw_afilt = sum_artifact_dicts(PFC_R_unrw_artifacts, HPC_R_unrw_artifacts)
            L_unrw_afilt = sum_artifact_dicts(PFC_L_unrw_artifacts, HPC_L_unrw_artifacts)

            PFC_R_rw_afilt = R_rw_afilt
            PFC_L_rw_afilt = L_rw_afilt
            PFC_R_unrw_afilt = R_unrw_afilt
            PFC_L_unrw_afilt = L_unrw_afilt

            BLA_R_rw_afilt = None
            BLA_L_rw_afilt = None
            BLA_R_unrw_afilt = None
            BLA_L_unrw_afilt = None

            HPC_R_rw_afilt = R_rw_afilt
            HPC_L_rw_afilt = L_rw_afilt
            HPC_R_unrw_afilt = R_unrw_afilt
            HPC_L_unrw_afilt = L_unrw_afilt

            # with open('sum_artifact_test.txt', 'w') as sys.stdout:
            #     print('PFC\n')
            #     print(PFC_R_rw_artifacts)
            #     for rat in PFC_R_rw_artifacts:
            #         for i, trial in enumerate(PFC_R_rw_artifacts[rat]):
            #             if True in trial:
            #                 print(f'trial_{i} has artifacts')
            #             else:
            #                 print(f'trial_{i} has no artifacts')
            #
            #     print('\n==========================\n')
            #     print('HPC\n')
            #     print(HPC_R_rw_artifacts)
            #     for rat in HPC_R_rw_artifacts:
            #         for i, trial in enumerate(HPC_R_rw_artifacts[rat]):
            #             if True in trial:
            #                 print(f'trial_{i} has artifacts')
            #             else:
            #                 print(f'trial_{i} has no artifacts')
            #
            #     print('\n==========================\n')
            #     print('Sum: PFC + HPC\n')
            #     print(R_rw_afilt)
            #     for rat in R_rw_afilt:
            #         for i, trial in enumerate(R_rw_afilt[rat]):
            #             if True in trial:
            #                 print(f'trial_{i} has artifacts')
            #             else:
            #                 print(f'trial_{i} has no artifacts')

        elif args.regions.lower() == 'pfc-bla':
            R_rw_afilt = None
            L_rw_afilt = sum_artifact_dicts(PFC_L_rw_artifacts, BLA_L_rw_artifacts)
            R_unrw_afilt = None
            L_unrw_afilt = sum_artifact_dicts(PFC_L_unrw_artifacts, BLA_L_unrw_artifacts)

            PFC_R_rw_afilt = R_rw_afilt
            PFC_L_rw_afilt = L_rw_afilt
            PFC_R_unrw_afilt = R_unrw_afilt
            PFC_L_unrw_afilt = L_unrw_afilt

            BLA_R_rw_afilt = R_rw_afilt
            BLA_L_rw_afilt = L_rw_afilt
            BLA_R_unrw_afilt = R_unrw_afilt
            BLA_L_unrw_afilt = L_unrw_afilt

            HPC_R_rw_afilt = None
            HPC_L_rw_afilt = None
            HPC_R_unrw_afilt = None
            HPC_L_unrw_afilt = None

        elif args.regions.lower() == 'bla-hpc':
            R_rw_afilt = None
            L_rw_afilt = sum_artifact_dicts(HPC_L_rw_artifacts, BLA_L_rw_artifacts)
            R_unrw_afilt = None
            L_unrw_afilt = sum_artifact_dicts(HPC_L_unrw_artifacts, BLA_L_unrw_artifacts)

            PFC_R_rw_afilt = None
            PFC_L_rw_afilt = None
            PFC_R_unrw_afilt = None
            PFC_L_unrw_afilt = None

            BLA_R_rw_afilt = R_rw_afilt
            BLA_L_rw_afilt = L_rw_afilt
            BLA_R_unrw_afilt = R_unrw_afilt
            BLA_L_unrw_afilt = L_unrw_afilt

            HPC_R_rw_afilt = R_rw_afilt
            HPC_L_rw_afilt = L_rw_afilt
            HPC_R_unrw_afilt = R_unrw_afilt
            HPC_L_unrw_afilt = L_unrw_afilt

        elif args.regions.lower() == 'pfc-pfc':
            R_rw_afilt = sum_artifact_dicts(PFC_R_rw_artifacts, PFC_L_rw_artifacts)
            L_rw_afilt = R_rw_afilt
            R_unrw_afilt = sum_artifact_dicts(PFC_R_unrw_artifacts, PFC_R_unrw_artifacts)
            L_unrw_afilt = R_unrw_afilt

            PFC_R_rw_afilt = R_rw_afilt
            PFC_L_rw_afilt = L_rw_afilt
            PFC_R_unrw_afilt = R_unrw_afilt
            PFC_L_unrw_afilt = L_unrw_afilt

            BLA_R_rw_afilt = None
            BLA_L_rw_afilt = None
            BLA_R_unrw_afilt = None
            BLA_L_unrw_afilt = None

            HPC_R_rw_afilt = None
            HPC_L_rw_afilt = None
            HPC_R_unrw_afilt = None
            HPC_L_unrw_afilt = None

        elif args.regions.lower() == 'hpc-hpc':      # not sure if i'll need this but i'm here so i might as well include it
            R_rw_afilt = sum_artifact_dicts(HPC_R_rw_artifacts, HPC_L_rw_artifacts)
            L_rw_afilt = R_rw_afilt
            R_unrw_afilt = sum_artifact_dicts(HPC_R_unrw_artifacts, HPC_L_unrw_artifacts)
            L_unrw_afilt = R_unrw_afilt

            PFC_R_rw_afilt = None
            PFC_L_rw_afilt = None
            PFC_R_unrw_afilt = None
            PFC_L_unrw_afilt = None

            BLA_R_rw_afilt = None
            BLA_L_rw_afilt = None
            BLA_R_unrw_afilt = None
            BLA_L_unrw_afilt = None

            HPC_R_rw_afilt = R_rw_afilt
            HPC_L_rw_afilt = L_rw_afilt
            HPC_R_unrw_afilt = R_unrw_afilt
            HPC_L_unrw_afilt = L_unrw_afilt



        # Ok now the final step is to use the artifact filters to remove noisy trials
        # again, before I expected a list of lists, but no we have a dictionary of list of lists.
        # Let's write a new function that accounts for this so we can use as much of what we already wrote.

        # with open('SHAPE_artifacts_test.txt', 'w') as sys.stdout:
        #     keys = list(PFC_R_rw_artifacts.keys())
        #     print(PFC_R_rw_artifacts)
        #     print('\n')
        #     print(keys[0])
        #     print(PFC_R_rw_afilt[keys[0]])
        #     print(f'len: {len(PFC_R_rw_afilt[keys[0]])}')
        #     print(f'shape:{np.shape(PFC_R_rw_afilt[keys[0]])}')
        #     print('\n====================\n')
        #     print(PFC_Rside_rewarded)
        #     print('\n')
        #     print(keys[0])
        #     print(PFC_Rside_rewarded[keys[0]])
        #     print(f'len: {len(PFC_Rside_rewarded[keys[0]])}')
        #     print(f'shape: {np.shape(PFC_Rside_rewarded[keys[0]])}')

        def filter_artifacts(trials_dict, artifacts_dict):
            '''
            Takes the artifacts identified in the artifacts_dict and removes from
            the trials_dict, any trial containing an artifact at any point in the
            trial. Because the artifacts_dict was built from the trials_dict (for
            each animal for each region) we expect the keys to be the same.
            '''
            assert list(trials_dict.keys()) == list(artifacts_dict.keys())
            rats = list(trials_dict.keys())

            filtered_trials = {}
            for rat in rats:
                has_artifact = []

                for afilt in artifacts_dict[rat]:
                    if True in afilt:
                        has_artifact.append(True)                   # indices of True will become dict keys, this way,
                                                                    # since all regions started out with the all trials,
                    if True not in afilt:                           # I can tell if the same trial was good in one region
                        has_artifact.append(False)                  # but not another. That is, the index 4 describes the 5th
                                                                    # trial in all regions, regardless if it was rejected during
                has_no_artifact = [not e for e in has_artifact]     # artifact detection in some but not other regions.
                                                                    # i.e. say we kept [0, 1, 2, 4, 10] from PFC
                # keys, values                                      # but only [0, 4, 10] from BLA. we can use the key, 4,
                i_keys = [i for i, b in enumerate(has_no_artifact) if b == True]    # to access the data for the same trial
                trials_no_artifact = np.array(trials_dict[rat])[has_no_artifact]    # from their respective regions, even though
                                                                                    # 4 is the 4th index from PFC and the 2nd index from BLA.
                # toss into dict to keep track of which trials were removed (by visual curation)
                filtered_trials[rat] = dict(zip(i_keys, trials_no_artifact))

            return filtered_trials


        PFC_R_rw_cleaned = filter_artifacts(PFC_Rside_rewarded, PFC_R_rw_artifacts)

        # ok I only want to have to do this once because once I get a good looking set of
        # trials, ideally I would perform remaining analyses only on the same set.
        # But in order to do that, I must first keep track of which trial I removed from
        # each region because the trials removed for any multi regional analysis must be
        # consistent across regions. (But for single region analyses, we take as much as
        # we can get)

        PFC_R_rw_cleaned = filter_artifacts(PFC_Rside_rewarded, PFC_R_rw_artifacts)
        PFC_L_rw_cleaned = filter_artifacts(PFC_Lside_rewarded, PFC_L_rw_artifacts)
        PFC_R_unrw_cleaned = filter_artifacts(PFC_Rside_unrewarded, PFC_R_unrw_artifacts)
        PFC_L_unrw_cleaned = filter_artifacts(PFC_Lside_unrewarded, PFC_L_unrw_artifacts)

        BLA_R_rw_cleaned = filter_artifacts(BLA_Rside_rewarded, BLA_R_rw_artifacts)
        BLA_L_rw_cleaned = filter_artifacts(BLA_Lside_rewarded, BLA_L_rw_artifacts)
        BLA_R_unrw_cleaned = filter_artifacts(BLA_Rside_unrewarded, BLA_R_unrw_artifacts)
        BLA_L_unrw_cleaned = filter_artifacts(BLA_Lside_unrewarded, BLA_L_unrw_artifacts)

        HPC_R_rw_cleaned = filter_artifacts(HPC_Rside_rewarded, HPC_R_rw_artifacts)
        HPC_L_rw_cleaned = filter_artifacts(HPC_Lside_rewarded, HPC_L_rw_artifacts)
        HPC_R_unrw_cleaned = filter_artifacts(HPC_Rside_unrewarded, HPC_R_unrw_artifacts)
        HPC_L_unrw_cleaned = filter_artifacts(HPC_Lside_unrewarded, HPC_L_unrw_artifacts)

        # ok great. now it's time visually curate clean looking trials.
        df_summary = pd.DataFrame()

        r = [
        'PFC_R_rw_cleaned', 'PFC_L_rw_cleaned', 'PFC_R_unrw_cleaned', 'PFC_L_unrw_cleaned',
        'BLA_R_rw_cleaned', 'BLA_L_rw_cleaned', 'BLA_R_unrw_cleaned', 'BLA_L_unrw_cleaned',
        'HPC_R_rw_cleaned', 'HPC_L_rw_cleaned', 'HPC_R_unrw_cleaned', 'HPC_L_unrw_cleaned'
        ]
        df_summary['region'] = r

        for key in list(PFC_R_rw_cleaned.keys()):
            ns = []

            ns.append(len(PFC_R_rw_cleaned[key])) if key in PFC_R_rw_cleaned.keys() else ns.append(0)
            ns.append(len(PFC_L_rw_cleaned[key])) if key in PFC_L_rw_cleaned.keys() else ns.append(0)
            ns.append(len(PFC_R_unrw_cleaned[key])) if key in PFC_R_unrw_cleaned.keys() else ns.append(0)
            ns.append(len(PFC_L_unrw_cleaned[key])) if key in PFC_L_unrw_cleaned.keys() else ns.append(0)

            ns.append(len(BLA_R_rw_cleaned[key])) if key in BLA_R_rw_cleaned.keys() else ns.append(0)
            ns.append(len(BLA_L_rw_cleaned[key])) if key in BLA_L_rw_cleaned.keys() else ns.append(0)
            ns.append(len(BLA_R_unrw_cleaned[key])) if key in BLA_R_unrw_cleaned.keys() else ns.append(0)
            ns.append(len(BLA_L_unrw_cleaned[key])) if key in BLA_L_unrw_cleaned.keys() else ns.append(0)

            ns.append(len(HPC_R_rw_cleaned[key])) if key in HPC_R_rw_cleaned.keys() else ns.append(0)
            ns.append(len(HPC_L_rw_cleaned[key])) if key in HPC_L_rw_cleaned.keys() else ns.append(0)
            ns.append(len(HPC_R_unrw_cleaned[key])) if key in HPC_R_unrw_cleaned.keys() else ns.append(0)
            ns.append(len(HPC_L_unrw_cleaned[key])) if key in HPC_L_unrw_cleaned.keys() else ns.append(0)

            df_summary[key] = ns

        df_summary['sum'] = np.zeros(len(df_summary))   # adding a last column which contains sum trial_ns
        for col in list(df_summary.columns)[1:-1]:
            df_summary['sum'] += df_summary[col]
        df_summary['sum'] = df_summary['sum'].astype(int)


        df_summary.to_csv('_'.join([args.global_labels, args.regions.upper(), 'RMS_FILTERED','trial_ns'])+'.csv')

        ##### TIME TO CURATE #####
        sys.stdout = sys.__stdout__

        print('\nCurate PFC:')

        curate_trials(PFC_R_rw_cleaned)
        curate_trials(PFC_L_rw_cleaned)
        curate_trials(PFC_R_unrw_cleaned)
        curate_trials(PFC_L_unrw_cleaned)

        print('\nCurate BLA')

        curate_trials(BLA_R_rw_cleaned)
        curate_trials(BLA_L_rw_cleaned)
        curate_trials(BLA_R_unrw_cleaned)
        curate_trials(BLA_L_unrw_cleaned)

        print('\nCurate HPC')

        curate_trials(HPC_R_rw_cleaned)
        curate_trials(HPC_L_rw_cleaned)
        curate_trials(HPC_R_unrw_cleaned)
        curate_trials(HPC_L_unrw_cleaned)

        print('\nCuration Complete!')

        fname = '_'.join([args.global_labels, args.regions.upper(), 'CURATED','trial_ns'])
        np.save('_'.join([fname, 'PFC_R_rw_curated.npy']), PFC_R_rw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'PFC_L_rw_curated.npy']), PFC_L_rw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'PFC_R_unrw_curated.npy']), PFC_R_unrw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'PFC_L_unrw_curated.npy']), PFC_L_unrw_cleaned, allow_pickle=True)

        np.save('_'.join([fname, 'BLA_R_rw_curated.npy']), BLA_R_rw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'BLA_L_rw_curated.npy']), BLA_L_rw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'BLA_R_unrw_curated.npy']), BLA_R_unrw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'BLA_L_unrw_curated.npy']), BLA_L_unrw_cleaned, allow_pickle=True)

        np.save('_'.join([fname, 'HPC_R_rw_curated.npy']), HPC_R_rw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'HPC_L_rw_curated.npy']), HPC_L_rw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'HPC_R_unrw_curated.npy']), HPC_R_unrw_cleaned, allow_pickle=True)
        np.save('_'.join([fname, 'HPC_L_unrw_curated.npy']), HPC_L_unrw_cleaned, allow_pickle=True)


        # now formatting a csv summarizing the artifact removal and curation process
        df_summary = pd.DataFrame()

        r = [
        'PFC_R_rw_cleaned', 'PFC_L_rw_cleaned', 'PFC_R_unrw_cleaned', 'PFC_L_unrw_cleaned',
        'BLA_R_rw_cleaned', 'BLA_L_rw_cleaned', 'BLA_R_unrw_cleaned', 'BLA_L_unrw_cleaned',
        'HPC_R_rw_cleaned', 'HPC_L_rw_cleaned', 'HPC_R_unrw_cleaned', 'HPC_L_unrw_cleaned'
        ]
        df_summary['region'] = r

        for key in list(PFC_R_rw_cleaned.keys()):
            ns = []

            ns.append(len(PFC_R_rw_cleaned[key])) if key in PFC_R_rw_cleaned.keys() else ns.append(0)
            ns.append(len(PFC_L_rw_cleaned[key])) if key in PFC_L_rw_cleaned.keys() else ns.append(0)
            ns.append(len(PFC_R_unrw_cleaned[key])) if key in PFC_R_unrw_cleaned.keys() else ns.append(0)
            ns.append(len(PFC_L_unrw_cleaned[key])) if key in PFC_L_unrw_cleaned.keys() else ns.append(0)

            ns.append(len(BLA_R_rw_cleaned[key])) if key in BLA_R_rw_cleaned.keys() else ns.append(0)
            ns.append(len(BLA_L_rw_cleaned[key])) if key in BLA_L_rw_cleaned.keys() else ns.append(0)
            ns.append(len(BLA_R_unrw_cleaned[key])) if key in BLA_R_unrw_cleaned.keys() else ns.append(0)
            ns.append(len(BLA_L_unrw_cleaned[key])) if key in BLA_L_unrw_cleaned.keys() else ns.append(0)

            ns.append(len(HPC_R_rw_cleaned[key])) if key in HPC_R_rw_cleaned.keys() else ns.append(0)
            ns.append(len(HPC_L_rw_cleaned[key])) if key in HPC_L_rw_cleaned.keys() else ns.append(0)
            ns.append(len(HPC_R_unrw_cleaned[key])) if key in HPC_R_unrw_cleaned.keys() else ns.append(0)
            ns.append(len(HPC_L_unrw_cleaned[key])) if key in HPC_L_unrw_cleaned.keys() else ns.append(0)

            df_summary[key] = ns

        df_summary['sum'] = np.zeros(len(df_summary))   # adding a last column which contains sum trial_ns
        for col in list(df_summary.columns)[1:-1]:
            df_summary['sum'] += df_summary[col]
        df_summary['sum'] = df_summary['sum'].astype(int)

        df_summary.to_csv('_'.join([args.global_labels, args.regions.upper(), 'CURATED','trial_ns'])+'.csv')


        sys.exit(1)






    if args.command == 'analyze':
        if args.global_labels == 'LDT_allrats':
            try:
                files = [args.path + f for f in os.listdir(args.path) if '._' not in f]

                file_dict_Veh = []
                file_dict_ABC = []
                for f in files:
                    d = np.load(f, allow_pickle=True).item()

                    if 'ChABC' in f:
                        file_dict_ABC.append(('_'.join(f.split('/')[-1].split('_')[-4:-1]), d))

                    if 'Vehicle' in f:
                        file_dict_Veh.append(('_'.join(f.split('/')[-1].split('_')[-4:-1]), d))

                Veh_DATA = dict(file_dict_Veh)
                ABC_DATA = dict(file_dict_ABC)

                assert(Veh_DATA.keys() == ABC_DATA.keys())

                DATA = {}
                for key in Veh_DATA:
                    DATA[key] = {**Veh_DATA[key], **ABC_DATA[key]}


            except FileNotFoundError as e:
                print('Unable to load in data. File not found.')
                sys.exit(1)
        else:
            try:
                files = [args.path + f for f in os.listdir(args.path) if '._' not in f]
                files = [f for f in files if '.npy' in f]
                file_dict = []
                for f in files:
                    d = np.load(f, allow_pickle=True).item()
                    file_dict.append(('_'.join(f.split('/')[-1].split('_')[-4:-1]), d))

                DATA = dict(file_dict)

            except FileNotFoundError as e:
                print('Unable to load in data. File not found.')
                sys.exit(1)


        RAT_ID = set.union(*[set(DATA[key].keys()) for key in DATA])

        print(RAT_ID)


        if not args.exclude is None:
            exclude = args.exclude.split(',')

            for rat in exclude:
                print(f'excluding {rat}')
                RAT_ID.discard(rat)

                for key in DATA:
                    if rat in DATA[key].keys():
                        del DATA[key][rat]

        RAT_ID = set.union(*[set(DATA[key].keys()) for key in DATA])
        RAT_ID = sorted(list(RAT_ID))       # just making this a list bc i think later we expect it to be a list
        print(f'Updated Rat_ID: {RAT_ID}')


        ###### ok getting everything out of our dictionaries.... ######
        ###### I probably would have preferred to do this with classes to make accessing thing a bit easier.... but whatever.
        for key in DATA:

            if 'PFC' in key:

                if '_rw' in key:
                    if '_R_' in key:
                        PFC_R_rw_cleaned = DATA[key]
                    if '_L_' in key:
                        PFC_L_rw_cleaned = DATA[key]

                if '_unrw' in key:
                    if '_R_' in key:
                        PFC_R_unrw_cleaned = DATA[key]
                    if '_L_' in key:
                        PFC_L_unrw_cleaned = DATA[key]

            if 'BLA' in key:

                if '_rw' in key:
                    if '_R_' in key:
                        BLA_R_rw_cleaned = DATA[key]
                    if '_L_' in key:
                        BLA_L_rw_cleaned = DATA[key]

                if '_unrw' in key:
                    if '_R_' in key:
                        BLA_R_unrw_cleaned = DATA[key]
                    if '_L_' in key:
                        BLA_L_unrw_cleaned = DATA[key]

            if 'HPC' in key:

                if '_rw' in key:
                    if '_R_' in key:
                        HPC_R_rw_cleaned = DATA[key]
                    if '_L_' in key:
                        HPC_L_rw_cleaned = DATA[key]

                if '_unrw' in key:
                    if '_R_' in key:
                        HPC_R_unrw_cleaned = DATA[key]
                    if '_L_' in key:
                        HPC_L_unrw_cleaned = DATA[key]


        # ok now we just have to account for merging all of our final curated sets per multi regional analysis.
        # as well as all of our weird spcecific use cases regarding rat exclusion.

        if 'Vehicle' in args.global_labels and 'HPC'.lower() in args.regions:
            assert 'VR5Ephys3_rat2' not in RAT_ID
            assert 'VR5Ephys3_rat2' not in set(HPC_R_rw_cleaned.keys())

        if 'VR5Ephys7_rat2' in RAT_ID and 'BLA'.lower() in args.regions:
            PFC_L_rw_cleaned['VR5Ephys7_rat2'] = PFC_R_rw_cleaned['VR5Ephys7_rat2']
            BLA_L_rw_cleaned['VR5Ephys7_rat2'] = BLA_R_rw_cleaned['VR5Ephys7_rat2']
            HPC_L_rw_cleaned['VR5Ephys7_rat2'] = HPC_R_rw_cleaned['VR5Ephys7_rat2']

            PFC_L_unrw_cleaned['VR5Ephys7_rat2'] = PFC_R_unrw_cleaned['VR5Ephys7_rat2']
            BLA_L_unrw_cleaned['VR5Ephys7_rat2'] = BLA_R_unrw_cleaned['VR5Ephys7_rat2']
            HPC_L_unrw_cleaned['VR5Ephys7_rat2'] = HPC_R_unrw_cleaned['VR5Ephys7_rat2']

        # if 'VR5Ephys7_rat2' in RAT_ID and args.regions.lower() == 'single':
        #     print(BLA_R_rw_cleaned.keys())
        #     BLA_L_rw_cleaned['VR5Ephys7_rat2'] = BLA_R_rw_cleaned['VR5Ephys7_rat2']
        #     BLA_L_unrw_cleaned['VR5Ephys7_rat2'] = BLA_R_unrw_cleaned['VR5Ephys7_rat2']

        def get_trialns_summary():
            df_summary = pd.DataFrame()

            r = [
            'PFC_R_rw_cleaned', 'PFC_L_rw_cleaned', 'PFC_R_unrw_cleaned', 'PFC_L_unrw_cleaned',
            'BLA_R_rw_cleaned', 'BLA_L_rw_cleaned', 'BLA_R_unrw_cleaned', 'BLA_L_unrw_cleaned',
            'HPC_R_rw_cleaned', 'HPC_L_rw_cleaned', 'HPC_R_unrw_cleaned', 'HPC_L_unrw_cleaned'
            ]
            df_summary['region'] = r

            for key in list(PFC_R_rw_cleaned.keys()):
                ns = []

                ns.append(len(PFC_R_rw_cleaned[key])) if key in PFC_R_rw_cleaned.keys() else ns.append(0)
                ns.append(len(PFC_L_rw_cleaned[key])) if key in PFC_L_rw_cleaned.keys() else ns.append(0)
                ns.append(len(PFC_R_unrw_cleaned[key])) if key in PFC_R_unrw_cleaned.keys() else ns.append(0)
                ns.append(len(PFC_L_unrw_cleaned[key])) if key in PFC_L_unrw_cleaned.keys() else ns.append(0)

                ns.append(len(BLA_R_rw_cleaned[key])) if key in BLA_R_rw_cleaned.keys() else ns.append(0)
                ns.append(len(BLA_L_rw_cleaned[key])) if key in BLA_L_rw_cleaned.keys() else ns.append(0)
                ns.append(len(BLA_R_unrw_cleaned[key])) if key in BLA_R_unrw_cleaned.keys() else ns.append(0)
                ns.append(len(BLA_L_unrw_cleaned[key])) if key in BLA_L_unrw_cleaned.keys() else ns.append(0)

                ns.append(len(HPC_R_rw_cleaned[key])) if key in HPC_R_rw_cleaned.keys() else ns.append(0)
                ns.append(len(HPC_L_rw_cleaned[key])) if key in HPC_L_rw_cleaned.keys() else ns.append(0)
                ns.append(len(HPC_R_unrw_cleaned[key])) if key in HPC_R_unrw_cleaned.keys() else ns.append(0)
                ns.append(len(HPC_L_unrw_cleaned[key])) if key in HPC_L_unrw_cleaned.keys() else ns.append(0)

                df_summary[key] = ns

            df_summary['sum'] = np.zeros(len(df_summary))   # adding a last column which contains sum trial_ns
            for col in list(df_summary.columns)[1:-1]:
                df_summary['sum'] += df_summary[col]
            df_summary['sum'] = df_summary['sum'].astype(int)

            return df_summary

        df_summary = get_trialns_summary()
        df_summary.to_csv('_'.join([args.global_labels, args.regions.upper(),'trial_ns'])+'.csv')


        def merge_trialn_keys(region_dict1, region_dict2):

            assert region_dict1.keys() == region_dict2.keys()    # just check that these are the same
            new_region_dict1 = {}
            new_region_dict2 = {}

            for rat in region_dict1:
                r1_trialn_keys = set(region_dict1[rat].keys())
                r2_trialn_keys = set(region_dict2[rat].keys())

                r1r2_trialn_keys = set.intersection(*[r1_trialn_keys, r2_trialn_keys])

                new_region_dict1[rat] = {}
                new_region_dict2[rat] = {}

                for key in r1r2_trialn_keys:
                    new_region_dict1[rat][key] = region_dict1[rat][key]
                    new_region_dict2[rat][key] = region_dict2[rat][key]


            return new_region_dict1, new_region_dict2

        def unpack_region_dict(region_dict):
            '''
            Takes a nested dictionary of trials and moves all trials in all dicts into a single list.
            Resulting list has no rat identification, order of rats should be preserved as
            dict.keys (which should be the same across all dicts) are sorted the same way every time.
            '''
            trials = []
            ratn_trialn = []

            if region_dict == {0:[0]}:
                trials = [0]
                ratn_trialn = 0
                return trials, ratn_trialn

            else:
                for rat in sorted(list(region_dict.keys())):
                    for trial_n in sorted(list(region_dict[rat])):
                        trials.append(region_dict[rat][trial_n])
                        ratn_trialn.append((rat, trial_n))

                return trials, ratn_trialn

        #['single', 'pfc-hpc', 'pfc-bla', 'bla-hpc', 'pfc-pfc', 'hpc-hpc']
        if args.regions.lower() != 'single':
            R1 = args.regions.split('-')[0]
            R2 = args.regions.split('-')[1]

            if R1.lower() == 'pfc' and R2.lower() == 'hpc':
                PFC_R_rw, HPC_R_rw = merge_trialn_keys(PFC_R_rw_cleaned, HPC_R_rw_cleaned)
                PFC_L_rw, HPC_L_rw = merge_trialn_keys(PFC_L_rw_cleaned, HPC_L_rw_cleaned)
                PFC_R_unrw, HPC_R_unrw = merge_trialn_keys(PFC_R_unrw_cleaned, HPC_R_unrw_cleaned)
                PFC_L_unrw, HPC_L_unrw = merge_trialn_keys(PFC_L_unrw_cleaned, HPC_L_unrw_cleaned)

                BLA_R_rw, BLA_L_rw, BLA_R_unrw, BLA_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

                for rat in RAT_ID:
                    assert PFC_R_rw[rat].keys() == HPC_R_rw[rat].keys()
                    assert PFC_L_rw[rat].keys() == HPC_L_rw[rat].keys()
                    assert PFC_R_unrw[rat].keys() == HPC_R_unrw[rat].keys()
                    assert PFC_L_unrw[rat].keys() == HPC_L_unrw[rat].keys()

            if R1.lower() == 'pfc' and R2.lower() == 'bla':
                PFC_R_rw, BLA_R_rw = ({0:[0]}, {0:[0]})
                PFC_L_rw, BLA_L_rw = merge_trialn_keys(PFC_L_rw_cleaned, BLA_L_rw_cleaned)
                PFC_R_unrw, BLA_R_unrw = ({0:[0]}, {0:[0]})
                PFC_L_unrw, BLA_L_unrw = merge_trialn_keys(PFC_L_unrw_cleaned, BLA_L_unrw_cleaned)

                HPC_R_rw, HPC_L_rw, HPC_R_unrw, HPC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

                for rat in RAT_ID:
                    #assert PFC_R_rw[rat].keys() == BLA_R_rw[rat].keys()
                    assert PFC_L_rw[rat].keys() == BLA_L_rw[rat].keys()
                    #assert PFC_R_unrw[rat].keys() == BLA_R_unrw[rat].keys()
                    assert PFC_L_unrw[rat].keys() == BLA_L_unrw[rat].keys()

            if R1.lower() == 'bla' and R2.lower() == 'hpc':
                BLA_R_rw, HPC_R_rw = ({0:[0]}, {0:[0]})
                BLA_L_rw, HPC_L_rw = merge_trialn_keys(BLA_L_rw_cleaned, HPC_R_rw_cleaned)
                BLA_R_unrw, HPC_R_unrw = ({0:[0]}, {0:[0]})
                BLA_L_unrw, HPC_L_unrw = merge_trialn_keys(BLA_L_unrw_cleaned, HPC_L_unrw_cleaned)

                for rat in RAT_ID:
                    #assert BLA_R_rw[rat].keys() == HPC_R_rw[rat].keys()
                    assert BLA_L_rw[rat].keys() == HPC_L_rw[rat].keys()
                    #assert BLA_R_unrw[rat].keys() == HPC_R_unrw[rat].keys()
                    assert BLA_L_unrw[rat].keys() == HPC_L_unrw[rat].keys()

                PFC_R_rw, PFC_L_rw, PFC_R_unrw, PFC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

            if R1.lower() == 'pfc' and R2.lower() == 'pfc':
                PFC_R_rw, PFC_L_rw = merge_trialn_keys(PFC_R_rw_cleaned, PFC_L_rw_cleaned)
                PFC_R_unrw, PFC_L_unrw = merge_trialn_keys(PFC_R_unrw_cleaned, PFC_L_unrw_cleaned)

                HPC_R_rw, HPC_L_rw, HPC_R_unrw, HPC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})
                BLA_R_rw, BLA_L_rw, BLA_R_unrw, BLA_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

                for rat in RAT_ID:
                    assert PFC_R_rw[rat].keys() == PFC_L_rw[rat].keys()
                    assert PFC_R_unrw[rat].keys() == PFC_L_unrw[rat].keys()

            if R1.lower() == 'pfc' and R2.lower() == 'pfc_r':
                PFC_R_rw, PFC_L_rw = merge_trialn_keys(PFC_R_rw_cleaned, PFC_L_rw_cleaned)
                PFC_R_unrw, PFC_L_unrw = merge_trialn_keys(PFC_R_unrw_cleaned, PFC_L_unrw_cleaned)

                HPC_R_rw, HPC_L_rw, HPC_R_unrw, HPC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})
                BLA_R_rw, BLA_L_rw, BLA_R_unrw, BLA_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

                for rat in RAT_ID:
                    assert PFC_R_rw[rat].keys() == PFC_L_rw[rat].keys()
                    assert PFC_R_unrw[rat].keys() == PFC_L_unrw[rat].keys()

            if R1.lower() == 'hpc' and R2.lower() == 'hpc':
                HPC_R_rw, HPC_L_rw = merge_trialn_keys(HPC_R_rw_cleaned, HPC_L_rw_cleaned)
                HPC_R_unrw, HPC_L_unrw = merge_trialn_keys(HPC_R_unrw_cleaned, HPC_L_unrw_cleaned)

                PFC_R_rw, PFC_L_rw, PFC_R_unrw, PFC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})
                BLA_R_rw, BLA_L_rw, BLA_R_unrw, BLA_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

                for rat in RAT_ID:
                    assert HPC_R_rw[rat].keys() == HPC_L_rw[rat].keys()
                    assert HPC_R_unrw[rat].keys() == HPC_L_unrw[rat].keys()

            print('regions merged.')

            if R1.lower() == 'within' and R2.lower() =='pfc':
                PFC_R_rw = PFC_R_rw_cleaned
                PFC_L_rw = PFC_L_rw_cleaned

                PFC_R_unrw = PFC_R_unrw_cleaned
                PFC_L_unrw = PFC_L_unrw_cleaned

                HPC_R_rw, HPC_L_rw, HPC_R_unrw, HPC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})
                BLA_R_rw, BLA_L_rw, BLA_R_unrw, BLA_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

            if R1.lower() == 'within' and R2.lower() =='hpc':
                HPC_R_rw = HPC_R_rw_cleaned
                HPC_L_rw = HPC_L_rw_cleaned

                HPC_R_unrw = HPC_R_unrw_cleaned
                HPC_L_unrw = HPC_L_unrw_cleaned

                PFC_R_rw, PFC_L_rw, PFC_R_unrw, PFC_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})
                BLA_R_rw, BLA_L_rw, BLA_R_unrw, BLA_L_unrw = ({0:[0]}, {0:[0]}, {0:[0]}, {0:[0]})

            def get_trialns_summary():
                df_summary = pd.DataFrame()

                r = [
                'PFC_R_rw', 'PFC_L_rw', 'PFC_R_unrw', 'PFC_L_unrw',
                'BLA_R_rw', 'BLA_L_rw', 'BLA_R_unrw', 'BLA_L_unrw',
                'HPC_R_rw', 'HPC_L_rw', 'HPC_R_unrw', 'HPC_L_unrw'
                ]
                df_summary['region'] = r

                for key in RAT_ID:
                    ns = []

                    ns.append(len(PFC_R_rw[key])) if key in PFC_R_rw.keys() else ns.append(0)
                    ns.append(len(PFC_L_rw[key])) if key in PFC_L_rw.keys() else ns.append(0)
                    ns.append(len(PFC_R_unrw[key])) if key in PFC_R_unrw.keys() else ns.append(0)
                    ns.append(len(PFC_L_unrw[key])) if key in PFC_L_unrw.keys() else ns.append(0)

                    ns.append(len(BLA_R_rw[key])) if key in BLA_R_rw.keys() else ns.append(0)
                    ns.append(len(BLA_L_rw[key])) if key in BLA_L_rw.keys() else ns.append(0)
                    ns.append(len(BLA_R_unrw[key])) if key in BLA_R_unrw.keys() else ns.append(0)
                    ns.append(len(BLA_L_unrw[key])) if key in BLA_L_unrw.keys() else ns.append(0)

                    ns.append(len(HPC_R_rw[key])) if key in HPC_R_rw.keys() else ns.append(0)
                    ns.append(len(HPC_L_rw[key])) if key in HPC_L_rw.keys() else ns.append(0)
                    ns.append(len(HPC_R_unrw[key])) if key in HPC_R_unrw.keys() else ns.append(0)
                    ns.append(len(HPC_L_unrw[key])) if key in HPC_L_unrw.keys() else ns.append(0)

                    df_summary[key] = ns

                df_summary['sum'] = np.zeros(len(df_summary))   # adding a last column which contains sum trial_ns
                for col in list(df_summary.columns)[1:-1]:
                    df_summary['sum'] += df_summary[col]
                df_summary['sum'] = df_summary['sum'].astype(int)

                return df_summary

            df_summary = get_trialns_summary()
            df_summary.to_csv('_'.join([args.global_labels, args.regions.upper(),'merged_trial_ns'])+'.csv')


        # let's handle all of our single region analyses first.

        # BEFORE THIS PART RUNS TAKE ALL TRIALS OUT OF A DICT AND INTO A LIST
        # or at least a list of lists.

        if args.regions.lower() == 'single':

            #### "unpacking" so to speak all of our region dictionaries so that each region
            #### is a single list of arrays (where each array is one trial from some animal)

            PFC_R_rw_cleaned, PFC_R_rw_rattrialn = unpack_region_dict(PFC_R_rw_cleaned)
            PFC_L_rw_cleaned, PFC_L_rw_rattrialn = unpack_region_dict(PFC_L_rw_cleaned)
            PFC_R_unrw_cleaned, PFC_R_unrw_rattrialn = unpack_region_dict(PFC_R_unrw_cleaned)
            PFC_L_unrw_cleaned, PFC_L_unrw_rattrialn = unpack_region_dict(PFC_L_unrw_cleaned)

            BLA_R_rw_cleaned, BLA_R_rw_rattrialn = unpack_region_dict(BLA_R_rw_cleaned)
            BLA_L_rw_cleaned, BLA_L_rw_rattrialn = unpack_region_dict(BLA_L_rw_cleaned)
            BLA_R_unrw_cleaned, BLA_R_unrw_rattrialn = unpack_region_dict(BLA_R_unrw_cleaned)
            BLA_L_unrw_cleaned, BLA_L_unrw_rattrialn = unpack_region_dict(BLA_L_unrw_cleaned)

            HPC_R_rw_cleaned, HPC_R_rw_rattrialn = unpack_region_dict(HPC_R_rw_cleaned)
            HPC_L_rw_cleaned, HPC_L_rw_rattrialn = unpack_region_dict(HPC_L_rw_cleaned)
            HPC_R_unrw_cleaned, HPC_R_unrw_rattrialn = unpack_region_dict(HPC_R_unrw_cleaned)
            HPC_L_unrw_cleaned, HPC_L_unrw_rattrialn = unpack_region_dict(HPC_L_unrw_cleaned)


            # toss all single region data into tuple
            PFC = (PFC_R_rw_cleaned, PFC_L_rw_cleaned, PFC_R_unrw_cleaned, PFC_L_unrw_cleaned)
            BLA = (BLA_R_rw_cleaned, BLA_L_rw_cleaned, BLA_R_unrw_cleaned, BLA_L_unrw_cleaned)
            HPC = (HPC_R_rw_cleaned, HPC_L_rw_cleaned, HPC_R_unrw_cleaned, HPC_L_unrw_cleaned)

            # identifying information for all single region analyses
            PFC_local_labels = ('PFC_Rside, Rewarded', 'PFC_Lside, Rewarded', 'PFC_Rside, Unrewarded', 'PFC_Lside, Unrewarded')
            BLA_local_labels = ('BLA_Rside, Rewarded', 'BLA_Lside, Rewarded', 'BLA_Rside, Unrewarded', 'BLA_Lside, Unrewarded')
            HPC_local_labels = ('HPC_Rside, Rewarded', 'HPC_Lside, Rewarded', 'HPC_Rside, Unrewarded', 'HPC_Lside, Unrewarded')

            if not args.quantify is None:

                # I'm only interested in the right side
                PFC = (PFC_R_rw_cleaned, PFC_R_unrw_cleaned)
                BLA = (BLA_R_rw_cleaned, BLA_R_unrw_cleaned)
                HPC = (HPC_R_rw_cleaned, HPC_R_unrw_cleaned)

                # identifying information for all single region analyses
                PFC_local_labels = ('PFC_Rside, Rewarded', 'PFC_Rside, Unrewarded')
                BLA_local_labels = ('BLA_Rside, Rewarded', 'BLA_Rside, Unrewarded')
                HPC_local_labels = ('HPC_Rside, Rewarded', 'HPC_Rside, Unrewarded')

                if args.quantify.lower() == 'spect':
                    epoch_len = 2
                    pre_bline = 0.1
                    freqs = np.arange(1,121)
                    n_dict = dict([(i, 3+(i-1)*9/(len(range(1,121))-1)) for i in range(1,121)]) # scale n with freq
                    f_range = (2,4)
                    t_range = (300,700)

                    PFC_R_rw_windowed_power, PFC_R_rw_avg_windowed_pow = windowed_pow_averages(PFC_R_rw_cleaned, f_range, t_range, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)
                    #PFC_L_rw_windowed_power, PFC_L_rw_avg_windowed_pow = windowed_pow_averages(PFC_L_rw_cleaned, f_range, t_range, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)

                    PFC_R_unrw_windowed_power, PFC_R_unrw_avg_windowed_pow = windowed_pow_averages(PFC_R_unrw_cleaned, f_range, t_range, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)
                    #PFC_L_unrw_windowed_power, PFC_L_unrw_avg_windowed_pow = windowed_pow_averages(PFC_L_unrw_cleaned, f_range, t_range, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)


                    def avg_windowed_pow_to_csv(avg_windowed_pow, rattrialn, csvfile):
                        avgs = []
                        rat_ns = []
                        trial_ns = []

                        for a, ratn_trialn in zip(avg_windowed_pow, rattrialn):
                            rat_n, trial_n = ratn_trialn
                            avgs.append(a)
                            rat_ns.append(rat_n)
                            trial_ns.append(trial_n)

                        df = pd.DataFrame()
                        df['rat_n'] = rat_ns
                        df['trial_n'] = trial_ns
                        df['average_power'] = avgs

                        df.to_csv(csvfile)

                    avg_windowed_pow_to_csv(PFC_R_rw_avg_windowed_pow, PFC_R_rw_rattrialn, '_'.join([args.global_labels, 'PFC_R_rw_avg_windowed_power', f'{f_range[0]}-{f_range[1]}Hz,{t_range[0]}-{t_range[1]}ms.csv']))
                    #avg_windowed_pow_to_csv(PFC_L_rw_avg_windowed_pow, PFC_L_rw_rattrialn, '_'.join([args.global_labels, 'PFC_L_rw_avg_windowed_power', '1-5Hz,200-800ms.csv']))
                    avg_windowed_pow_to_csv(PFC_R_unrw_avg_windowed_pow, PFC_R_unrw_rattrialn, '_'.join([args.global_labels, 'PFC_R_unrw_avg_windowed_power', f'{f_range[0]}-{f_range[1]}Hz,{t_range[0]}-{t_range[1]}ms.csv']))
                    #avg_windowed_pow_to_csv(PFC_R_unrw_avg_windowed_pow, PFC_L_unrw_rattrialn, '_'.join([args.global_labels, 'PFC_L_unrw_avg_windowed_power', '1-5Hz,200-800ms.csv']))

                    sys.exit(1)

                if args.quantify.lower() == 'itpc':
                    print('now quantifying ITPCs... ')
                    epoch_len = 2
                    pre_bline = 0.3
                    freqs = np.arange(4,40)

                    # PFC = (PFC_R_rw_cleaned, PFC_L_rw_cleaned, PFC_R_unrw_cleaned, PFC_L_unrw_cleaned) in in this order:
                    PFC_observed_ITPC = [ITPC(trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs) for trials in PFC]
                    HPC_observed_ITPC = [ITPC(trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs) for trials in HPC]

                    print('computed observed ITPCs. now beginning permutation testing...')

                    start = datetime.datetime.now()
                    print(f'PFC permutation testing began at {start}')

                    for i, x_trials in enumerate(PFC):
                        n_permutations = 10000
                        observed_ITPC = PFC_observed_ITPC[i]
                        local_label = PFC_local_labels[i]
                        fname = args.global_labels + ' ' + local_label + ' z-scored ITPC f=[4,40,1], t=-300-1000ms.npy'

                        ITPCz = get_zscored_ITPC(x_trials, n_permutations, observed_ITPC, args.s_pre, args.s_post, pre_bline, epoch_len, freqs)
                        np.save(fname, ITPCz, allow_pickle=True)

                    end = datetime.datetime.now()
                    print(f'PFC permutation testing ended at {end}')
                    print(f'duration: {end - start}\n')

                    start = datetime.datetime.now()
                    print(f'HPC permutation testing began at {start}')

                    for i, x_trials in enumerate(HPC):
                        n_permutations = 10000
                        observed_ITPC = HPC_observed_ITPC[i]
                        local_label = HPC_local_labels[i]
                        fname = args.global_labels + ' ' + local_label + ' z-scored ITPC f=[4,40,1], t=-300-1000ms.npy'

                        ITPCz = get_zscored_ITPC(x_trials, n_permutations, observed_ITPC, args.s_pre, args.s_post, pre_bline, epoch_len, freqs)
                        np.save(fname, ITPCz, allow_pickle=True)

                    end = datetime.datetime.now()
                    print(f'HPC permutation testing ended at {end}')
                    print(f'duration: {end - start}\n')

                    sys.exit()

            # ERP fn calls and plot
            PFC_ERP = (ERP(trials) for trials in PFC)
            BLA_ERP = (ERP(trials) for trials in BLA)
            HPC_ERP = (ERP(trials) for trials in HPC)

            # for erp, label in zip(PFC_ERP, PFC_local_labels):
            #     plot_ERP(erp, args.global_labels, label)
            #
            # for erp, label in zip(BLA_ERP, BLA_local_labels):
            #     plot_ERP(erp, args.global_labels, label)
            #
            # for erp, label in zip(HPC_ERP, HPC_local_labels):
            #     plot_ERP(erp, args.global_labels, label)

            '''
            # filtered signal traces, fn calls and plot
            epoch_len = 1
            pre_bline = 0.1
            lc = 10
            hc = 20
            fs = 2000
            order = 4

            PFC_filtered_sigs = (bp_filter_sigs(sig, lc, hc, fs, order, args.s_pre, args.s_post, epoch_len, pre_bline) for sig in PFC)

            for bp_sigs, label in zip(PFC_filtered_sigs, PFC_local_labels):
                plot_filtered_sigs(bp_sigs, lc, hc, epoch_len, pre_bline, args.global_labels, label)
            '''

            # SPECTROGRAM fn calls and plot
            epoch_len = 2
            pre_bline = 0.1
            freqs = np.arange(1,121)
            n_dict = dict([(i, 3+(i-1)*9/(len(range(1,121))-1)) for i in range(1,121)]) # scale n with freq

            PFC_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in PFC)
            #BLA_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in BLA)
            HPC_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in HPC)

            for spect, label in zip(PFC_spect_hi, PFC_local_labels):
                plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            #for spect, label in zip(BLA_spect_hi, BLA_local_labels):
            #    plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            for spect, label in zip(HPC_spect_hi, HPC_local_labels):
                plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)


            epoch_len = 2
            pre_bline = 0.1
            freqs = np.arange(1,41)
            n_dict = dict([(i, 3+(i-1)*3/(len(range(1,41))-1)) for i in range(1,41)]) # scale n with freq

            PFC_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in PFC)
            #BLA_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in BLA)
            HPC_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in HPC)

            for spect, label in zip(PFC_spect_hi, PFC_local_labels):
                plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            #for spect, label in zip(BLA_spect_hi, BLA_local_labels):
            #    plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            for spect, label in zip(HPC_spect_hi, HPC_local_labels):
                plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)


            epoch_len = 2
            pre_bline = 0.1
            freqs = np.arange(1,26)
            n_dict = dict([(i, 3+(i-1)*3/(len(range(1,26))-1)) for i in range(1,26)]) # scale n with freq

            PFC_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in PFC)
            #BLA_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in BLA)
            HPC_spect_hi = (spectrogram(trials, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline) for trials in HPC)

            for spect, label in zip(PFC_spect_hi, PFC_local_labels):
                plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            #for spect, label in zip(BLA_spect_hi, BLA_local_labels):
            #    plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            for spect, label in zip(HPC_spect_hi, HPC_local_labels):
                plot_spectrogram(spect, epoch_len, pre_bline, freqs, args.global_labels, label)

            sys.exit()



            # ITPC fn calls and plot
            epoch_len = 2
            pre_bline = 0.3
            freqs = np.arange(4,40)

            PFC_ITPC = (ITPC(trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs) for trials in PFC)
            BLA_ITPC = (ITPC(trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs) for trials in BLA)
            HPC_ITPC = (ITPC(trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs) for trials in HPC)

            for itpc, label in zip(PFC_ITPC, PFC_local_labels):
                plot_ITPCs(itpc, epoch_len, pre_bline, freqs, args.global_labels, label)

            for itpc, label in zip(BLA_ITPC, BLA_local_labels):
                plot_ITPCs(itpc, epoch_len, pre_bline, freqs, args.global_labels, label)

            for itpc, label in zip(HPC_ITPC, HPC_local_labels):
                plot_ITPCs(itpc, epoch_len, pre_bline, freqs, args.global_labels, label)


            sys.exit()



        PFC_R_rw_cleaned, PFC_R_rw_rattrialn = unpack_region_dict(PFC_R_rw)
        PFC_L_rw_cleaned, PFC_L_rw_rattrialn = unpack_region_dict(PFC_L_rw)
        PFC_R_unrw_cleaned, PFC_R_unrw_rattrialn = unpack_region_dict(PFC_R_unrw)
        PFC_L_unrw_cleaned, PFC_L_unrw_rattrialn = unpack_region_dict(PFC_L_unrw)

        BLA_R_rw_cleaned, BLA_R_rw_rattrialn = unpack_region_dict(BLA_R_rw)
        BLA_L_rw_cleaned, BLA_L_rw_rattrialn = unpack_region_dict(BLA_L_rw)
        BLA_R_unrw_cleaned, BLA_R_unrw_rattrialn = unpack_region_dict(BLA_R_unrw)
        BLA_L_unrw_cleaned, BLA_L_unrw_rattrialn = unpack_region_dict(BLA_L_unrw)

        HPC_R_rw_cleaned, HPC_R_rw_rattrialn = unpack_region_dict(HPC_R_rw)
        HPC_L_rw_cleaned, HPC_L_rw_rattrialn = unpack_region_dict(HPC_L_rw)
        HPC_R_unrw_cleaned, HPC_R_unrw_rattrialn = unpack_region_dict(HPC_R_unrw)
        HPC_L_unrw_cleaned, HPC_L_unrw_rattrialn = unpack_region_dict(HPC_L_unrw)

        # every combination of 2 args.regions (of course no bla-bla)
        if args.regions.lower() == 'pfc-hpc':

            if args.hemisphere:
                if args.hemisphere.lower() == 'r':
                    # for now just consider R side bc analyses are getting very long
                    # R1 = (PFC_R_rw_cleaned, PFC_L_rw_cleaned, PFC_R_unrw_cleaned, PFC_L_unrw_cleaned)
                    # R2 = (HPC_R_rw_cleaned, HPC_L_rw_cleaned, HPC_R_unrw_cleaned, HPC_L_unrw_cleaned)
                    # R1R2_labels = ['PFC-HPC_Rside Rewarded', 'PFC-HPC_Lside Rewarded', 'PFC-HPC_Rside Unrewarded', 'PFC-HPC_Lside Unrewarded']

                    R1 = (PFC_R_rw_cleaned, PFC_R_unrw_cleaned)
                    R2 = (HPC_R_rw_cleaned, HPC_R_unrw_cleaned)
                    R1R2_labels = ['PFC-HPC_Rside Rewarded', 'PFC-HPC_Rside Unrewarded']

                    R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
                    R1 = tuple([r for r in R1 if len(r) != 0])
                    R2 = tuple([r for r in R2 if len(r) != 0])

                if args.hemisphere.lower() == 'l':
                    R1 = (PFC_L_rw_cleaned, PFC_L_unrw_cleaned)
                    R2 = (HPC_L_rw_cleaned, HPC_L_unrw_cleaned)
                    R1R2_labels = ['PFC-HPC_Lside Rewarded', 'PFC-HPC_Lside Unrewarded']

                    R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
                    R1 = tuple([r for r in R1 if len(r) != 0])
                    R2 = tuple([r for r in R2 if len(r) != 0])

            elif not args.hemisphere:
                R1 = (PFC_R_rw_cleaned, PFC_L_rw_cleaned, PFC_R_unrw_cleaned, PFC_L_unrw_cleaned)
                R2 = (HPC_R_rw_cleaned, HPC_L_rw_cleaned, HPC_R_unrw_cleaned, HPC_L_unrw_cleaned)
                R1R2_labels = ['PFC-HPC_Rside Rewarded', 'PFC-HPC_Lside Rewarded', 'PFC-HPC_Rside Unrewarded','PFC-HPC_Lside Unrewarded']

                R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
                R1 = tuple([r for r in R1 if len(r) != 0])
                R2 = tuple([r for r in R2 if len(r) != 0])


        elif args.regions.lower() == 'pfc-bla':
            R1 = (PFC_L_rw_cleaned, PFC_L_unrw_cleaned)
            R2 = (BLA_L_rw_cleaned, BLA_L_unrw_cleaned)
            R1R2_labels = ['PFC-BLA_Lside Rewarded', 'PFC-BLA_Lside Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        elif args.regions.lower() == 'bla-hpc':
            R1 = (BLA_L_rw_cleaned, BLA_L_unrw_cleaned)
            R2 = (HPC_L_rw_cleaned, HPC_L_unrw_cleaned)
            R1R2_labels = ['BLA-HPC_Lside Rewarded', 'BLA-HPC_Lside Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        elif args.regions.lower() == 'pfc-pfc':
            R1 = (PFC_R_rw_cleaned, PFC_R_unrw_cleaned)
            R2 = (PFC_L_rw_cleaned, PFC_L_unrw_cleaned)
            R1R2_labels = ['PFC(R)-PFC(L) Rewarded', 'PFC(R)-PFC(L) Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        elif args.regions.lower() == 'pfc-pfc_r':
            # added _r for reverse order
            R1 = (PFC_L_rw_cleaned, PFC_L_unrw_cleaned)
            R2 = (PFC_R_rw_cleaned, PFC_R_unrw_cleaned)
            R1R2_labels = ['PFC(L)-PFC(R) Rewarded', 'PFC(L)-PFC(R) Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        elif args.regions.lower() == 'hpc-hpc':
            R1 = (HPC_R_rw_cleaned, HPC_R_unrw_cleaned)
            R2 = (HPC_L_rw_cleaned, HPC_L_unrw_cleaned)
            R1R2_labels = ['HPC(R)-HPC(L) Rewarded', 'HPC(R)-HPC(L) Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        elif args.regions.lower() == 'within-pfc':
            R1 = (PFC_R_rw_cleaned, PFC_R_unrw_cleaned)
            R2 = (PFC_R_rw_cleaned, PFC_R_unrw_cleaned)
            R1R2_labels = ['PFC(R)-PFC(R) Rewarded', 'PFC(R)-PFC(R) Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        elif args.regions.lower() == 'within-hpc':
            R1 = (HPC_R_rw_cleaned, HPC_R_unrw_cleaned)
            R2 = (HPC_R_rw_cleaned, HPC_R_unrw_cleaned)
            R1R2_labels = ['HPC(R)-HPC(R) Rewarded', 'HPC(R)-HPC(R) Unrewarded']

            R1R2_labels = [R1R2_labels[i] for i, r in enumerate(R1) if len(r) != 0]
            R1 = tuple([r for r in R1 if len(r) != 0])
            R2 = tuple([r for r in R2 if len(r) != 0])

        if not args.regions == 'single':
            if not None in R1 and not None in R2:   # this is old: need to update this check

                if not args.quantify is None:
                    if args.quantify.lower() == 'pli':

                        epoch_len = 2
                        pre_bline = 0.3
                        freqs = np.arange(4,41)
                        ns = [6 + (i-4)*4/len(range(4,40)) for i in range(4,41)]

                        observed_PLIs = [PLI(R1_trials, R2_trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs, ns) for R1_trials, R2_trials in zip(R1, R2)]

                        print('computed observed PLIs. now beginning permutation testing...')

                        start = datetime.datetime.now()
                        print(f'PFC permutation testing began at {start}')

                        for i, (x_trials, y_trials) in enumerate(zip(R1, R2)):
                            n_permutations = 10000
                            observed_PLI = observed_PLIs[i]
                            local_label = R1R2_labels[i]
                            fname = args.global_labels + ' ' + local_label + ' z-scored PLI f=[4,40,1], t=-300-1000ms.npy'

                            PLIz = get_zscored_PLI(x_trials, y_trials, n_permutations, observed_PLI, args.s_pre, args.s_post, pre_bline, epoch_len, freqs, ns)
                            print(np.shape(PLIz))
                            np.save(fname, PLIz, allow_pickle=True)

                        end = datetime.datetime.now()
                        print(f'{args.regions.upper()} permutation testing ended at {end}')
                        print(f'duration: {end - start}\n')

                    # PACz
                    if args.quantify.lower() == 'pacz':
                        print('\ncomputing PACs to zscore...')
                        epoch_len = 2
                        pre_bline = 0.1

                        freqs4pha = np.arange(3,13.5,0.5)
                        freqs4pow = np.arange(20,121)

                        t_start = 0.3
                        t_end = 0.6

                        R1_R_rw_cleaned, R1_R_unrw_cleaned = R1
                        R2_R_rw_cleaned, R2_R_unrw_cleaned = R2

                        rw_PACz = PACz_wrapped(R2_R_rw_cleaned, R1_R_rw_cleaned, freqs4pha, freqs4pow, args.s_pre, args.s_post, epoch_len, pre_bline, t_start, t_end)
                        np.save(args.global_labels + ' ' + R1R2_labels[0] + ' pha=[3,13.5,0.5], pwr=[20,121,1], t=300-600ms PACz RAW POWER.npy', rw_PACz)

                        unrw_PACz = PACz_wrapped(R2_R_unrw_cleaned, R1_R_unrw_cleaned, freqs4pha, freqs4pow, args.s_pre, args.s_post, epoch_len, pre_bline, t_start, t_end)
                        np.save(args.global_labels + ' ' + R1R2_labels[1] + ' pha=[3,13.5,0.5], pwr=[20,121,1], t=300-600ms PACz RAW POWER.npy', unrw_PACz)

                        plt.figure(figsize=(5,5))
                        cs = plt.contourf(freqs4pha, freqs4pow, rw_PACz, levels=np.linspace(1.96,2.9), extend='min', cmap='jet')
                        cbar = plt.colorbar(cs, ticks=np.arange(-2,3.4,0.4),shrink=0.9)
                        plt.xlabel('Frequency for phase (Hz)')
                        plt.ylabel('Frequency for power (Hz)')
                        plt.title(args.global_labels + ' ' + R1R2_labels[0]+ ' Z-scored PAC')
                        plt.show()

                        plt.figure(figsize=(5,5))
                        cs = plt.contourf(freqs4pha, freqs4pow, unrw_PACz, levels=np.linspace(1.96,2.9), extend='min', cmap='jet')
                        cbar = plt.colorbar(cs, ticks=np.arange(-2,3.4,0.4),shrink=0.9)
                        plt.xlabel('Frequency for phase (Hz)')
                        plt.ylabel('Frequency for power (Hz)')
                        plt.title(args.global_labels + ' ' + R1R2_labels[1]+ ' Z-scored PAC')
                        plt.show()

                    else:
                        print('please specify metric to quantify')

                    sys.exit()

                # ISPC
                if not args.analysis or args.analysis == 'IPSC':
                    print('\ncomputing ISPCs...')
                    epoch_len = 2
                    pre_bline = 0.3
                    freqs = np.arange(4,41)
                    ns = [6 + (i-4)*4/len(range(4,40)) for i in range(4,41)]

                    ISPCs = (ISPC(R1_trials, R2_trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs, ns) for R1_trials, R2_trials in zip(R1, R2))

                    print('plotting ISPCs...\n')
                    for ispc, label in zip(ISPCs, R1R2_labels):
                        plot_ISPCs(ispc, epoch_len, pre_bline, freqs, args.global_labels, label)

                # PLI
                if not args.analysis or args.analysis == 'PLI':
                    print('\ncomputing PLIs...')
                    epoch_len = 2
                    pre_bline = 0.3
                    freqs = np.arange(4,41)
                    ns = [6 + (i-4)*4/len(range(4,40)) for i in range(4,41)]

                    PLIs = (PLI(R1_trials, R2_trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs, ns) for R1_trials, R2_trials in zip(R1, R2))

                    print('plotting PLIs...\n')
                    for pli, label in zip(PLIs, R1R2_labels):
                        plot_PLIs(pli, epoch_len, pre_bline, freqs, args.global_labels, label)

                # dPLI
                if not args.analysis or args.analysis == 'dPLI':
                    print('\ncomputing dPLIs...')
                    epoch_len = 2
                    pre_bline = 0.3
                    freqs = np.arange(4,41)
                    ns = [6 + (i-4)*4/len(range(4,40)) for i in range(4,41)]

                    dPLIs = (dPLI(R1_trials, R2_trials, args.s_pre, args.s_post, pre_bline, epoch_len, freqs, ns) for R1_trials, R2_trials in zip(R1, R2))

                    print('plotting dPLIs...\n')
                    for dpli, label in zip(dPLIs, R1R2_labels):
                        plot_dPLIs(dpli, epoch_len, pre_bline, freqs, args.global_labels, label)

                # Phase-Amp Coupling (not Z scored)
                # Plotting power as a distribution over phase angles (pi/30)
                if not args.analysis or args.analysis == 'PAC':
                    print('\ncomputing PACs...')
                    epoch_len = 2
                    pre_bline = 0.1
                    t0 = 200
                    t1 = 600

                    #composite_signals = [get_composite_signal_PAC_dist(R2_trials, R1_trials, args.s_pre, args.s_post, epoch_len, pre_bline, freqs) for R1_trials, R2_trials in zip(R1,R2)
                    assert len(R1) == len(R2)
                    assert len(R1R2_labels) == len(R2)

                    for i in range(len(R1)):
                        R1_trial = R1[i]
                        R2_trial = R2[i]
                        label = R1R2_labels[i]

                        pha_bins, norm_pow, pac = get_composite_signal_PAC_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1)
                        plot_avg_pwr_by_bin(pha_bins, norm_pow, args.global_labels, label, t0, t1, pac)
                        plot_avg_pwr_by_bin_polar(pha_bins, norm_pow, args.global_labels, label, t0, t1, pac)

                    # t0=400
                    # t1=800
                    # for i in range(len(R1)):
                    #     R1_trial = R1[i]
                    #     R2_trial = R2[i]
                    #     label = R1R2_labels[i]
                    #
                    #     pha_bins, norm_pow, pac = get_composite_signal_PAC_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1)
                    #     plot_avg_pwr_by_bin(pha_bins, norm_pow, args.global_labels, label, t0, t1, pac)

                    # t0=0
                    # t1=800
                    # for i in range(len(R1)):
                    #     R1_trial = R1[i]
                    #     R2_trial = R2[i]
                    #     label = R1R2_labels[i]
                    #
                    #     pha_bins, norm_pow, pac = get_composite_signal_PAC_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1)
                    #     plot_avg_pwr_by_bin(pha_bins, norm_pow, args.global_labels, label, t0, t1, pac)


                    # PACs = (doubleplot(average_nbym_matrices(pow_by_pha(pow, pha_bins, nfs=101, nbins=100))) for pha_bins, pow in composite_signals)
                    #
                    # print('plotting PACs...\n')
                    # for PAC, label in zip(PACs, R1R2_labels):
                    #     plot_PAC(PAC, args.global_labels, label, args.regions)

                # Phase-Amp Coupling (not Z scored)
                if not args.analysis or args.analysis == 'PACzdist':
                    print('\ncomputing PACs...')
                    epoch_len = 2
                    pre_bline = 0.1
                    t0 = 0
                    t1 = 400

                    #composite_signals = [get_composite_signal_PAC_dist(R2_trials, R1_trials, args.s_pre, args.s_post, epoch_len, pre_bline, freqs) for R1_trials, R2_trials in zip(R1,R2)
                    assert len(R1) == len(R2)
                    assert len(R1R2_labels) == len(R2)

                    for i in range(len(R1)):
                        R1_trial = R1[i]
                        R2_trial = R2[i]
                        label = R1R2_labels[i]

                        zscore_PAC_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)
                        #zscore_MI_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)

                    t0=400
                    t1=800
                    for i in range(len(R1)):
                        R1_trial = R1[i]
                        R2_trial = R2[i]
                        label = R1R2_labels[i]

                        zscore_PAC_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)
                        #zscore_MI_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)

                    t0=0
                    t1=800
                    for i in range(len(R1)):
                        R1_trial = R1[i]
                        R2_trial = R2[i]
                        label = R1R2_labels[i]

                        #zscore_PAC_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)
                        #zscore_PAC_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)
                # Phase-Amp Coupling (not Z scored)

                if not args.analysis or args.analysis == 'MIzdist':
                    print('\ncomputing PACs...')
                    epoch_len = 2
                    pre_bline = 0.1
                    t0 = 200
                    t1 = 600

                    #composite_signals = [get_composite_signal_PAC_dist(R2_trials, R1_trials, args.s_pre, args.s_post, epoch_len, pre_bline, freqs) for R1_trials, R2_trials in zip(R1,R2)
                    assert len(R1) == len(R2)
                    assert len(R1R2_labels) == len(R2)

                    for i in range(len(R1)):
                        R1_trial = R1[i]
                        R2_trial = R2[i]
                        label = R1R2_labels[i]

                        # looking for unrewarded trials only for inter hemispheric PAC in the PFC
                        if ' Rewarded' in label:
                            continue

                        print(f'\n{label} {t0}-{t1}: ')
                        #zscore_PAC_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)
                        zscore_MI_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)

                    # t0=400
                    # t1=800
                    # for i in range(len(R1)):
                    #     R1_trial = R1[i]
                    #     R2_trial = R2[i]
                    #     label = R1R2_labels[i]
                    #
                    #     if ' Unrewarded' in label:
                    #         continue
                    #
                    #     print(f'\n{label} {t0}-{t1}: ')
                    #     #zscore_PAC_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)
                    #     zscore_MI_pwrpha_dist(R2_trial, R1_trial, args.s_pre, args.s_post, epoch_len, pre_bline, t0, t1, args.global_labels, label)

                if not args.analysis or args.analysis == 'PACcm':

                    for i in range(len(R1)):
                        R1_trials = R1[i]
                        R2_trials = R2[i]
                        fs4pha = np.around(np.arange(3,13,0.1), 1)
                        fs4pow = np.arange(20,121,1)
                        t_win = (0.0, 0.4)
                        global_labels = args.global_labels
                        local_label = R1R2_labels[i]
                        epoch_len = 2
                        s_pre = args.s_pre
                        pre_bline = 0.1
                        r = args.regions

                        PACcm(R1_trials, R2_trials, fs4pha, fs4pow, t_win, global_labels, local_label, epoch_len, s_pre, pre_bline, r)



                    # R1_trials = R1[1]
                    # R2_trials = R1[1]               # =========== PFC-PFC SAME SITE =========== #
                    # print(np.shape(R1_trials))
                    # print(np.shape(R2_trials))
                    # assert len(R1_trials) == len(R2_trials)
                    # trial_n = len(R1_trials)
                    #
                    # print('\ncompute PACs for raw comodulogram...')
                    # epoch_len = 2
                    # pre_bline = 0.1
                    #
                    # fs = 2000
                    # epoch_samples = epoch_len * fs
                    # epoch_start = int((args.s_pre * fs ) - (epoch_samples / 2))          # slices symmetrically around timestamps
                    # epoch_end = int(epoch_samples) + epoch_start
                    # baseline_end = int((epoch_samples / 2) - (pre_bline * fs))
                    #
                    # def get_PAC(pha, pwr):
                    #     PAC = abs(np.mean(pwr * np.exp(1j * pha)))
                    #     return(PAC)
                    #
                    # pac_by_phase = []
                    # for f_pha in np.arange(3,13,0.1):
                    #
                    #     pac_by_power = []
                    #     for f_pow in np.arange(20,121):
                    #         power = [abs(compute_mwt(trial, fs=2000, peak_freq=f_pow, n=12)) ** 2 for trial in R1_trials]     # high n for high f precision
                    #         power = [trial[epoch_start:epoch_end] for trial in power]
                    #         norm_power = [np.array(trial) / np.average(trial[:baseline_end]) for trial in power]
                    #         norm_power = [trial[2800:3600] for trial in norm_power]
                    #         pwr = np.reshape(norm_power, (1, trial_n*800))[0]
                    #
                    #
                    #         phase = [np.angle(compute_mwt(trial, fs=2000, peak_freq=f_pha, n=12)) for trial in R2_trials]      # high n for high f precision
                    #         phase = [trial[epoch_start:epoch_end] for trial in phase]
                    #         phase = [trial[2800:3600] for trial in phase]
                    #         pha = np.reshape(phase, (1,trial_n*800))[0]
                    #
                    #         pac = get_PAC(pha, pwr)
                    #
                    #         pac_by_power.append(pac)
                    #     pac_by_phase.append(pac_by_power)
                    #
                    # pac_2d = np.array(pac_by_phase).T
                    #
                    # freqs4pha = np.arange(3,13,0.1)
                    # freqs4pow = np.arange(20,121)
                    # levels = np.linspace(np.min(pac_2d), np.max(pac_2d),50)
                    #
                    # plt.figure(figsize=(5,5))
                    # cs = plt.contourf(freqs4pha, freqs4pow, pac_2d, levels=levels, cmap='jet')
                    # cbar = plt.colorbar(cs,shrink=0.9)
                    #
                    # title =
                    # plt.xlabel('Frequency for phase (Hz)')
                    # plt.ylabel('Frequency for power (Hz)')
                    #
                    # plt.show()

                if not args.analysis or args.analysis == 'PACzcm':
                    n_perm = 10000
                    epoch_len = 2
                    pre_bline = 0.1
                    fs4pha = np.around(np.arange(3,13,0.2), 1)
                    fs4pow = np.arange(20,101,2)
                    global_labels = args.global_labels
                    s_pre = args.s_pre

                    with open('PACz cm params.txt', 'w') as f:
                        lines = [
                        'n_perm = 5000\n',
                        f'epoch_len = {epoch_len}\n',
                        f'pre_bline = {pre_bline}\n',
                        'fs4pha = np.around(np.arange(3,13,0.2), 1)\n',
                        f'{fs4pha}\n'
                        'fs4pow = np.arange(20,101,2)\n',
                        f'{fs4pow}\n'
                        f'global_labels = {args.global_labels}\n',
                        f's_pre = {args.s_pre}\n'
                        ]
                        f.writelines(lines)


                    t_start = 0
                    t_end = 0.4
                    PACz_cm(R1[0], R2[0], R1R2_labels[0], n_perm, epoch_len, pre_bline, fs4pha, fs4pow, t_start, t_end, global_labels, s_pre)
                    PACz_cm(R1[1], R2[1], R1R2_labels[1], n_perm, epoch_len, pre_bline, fs4pha, fs4pow, t_start, t_end, global_labels, s_pre)

                    t_start = 0.4
                    t_end = 0.8
                    PACz_cm(R1[0], R2[0], R1R2_labels[0], n_perm, epoch_len, pre_bline, fs4pha, fs4pow, t_start, t_end, global_labels, s_pre)
                    PACz_cm(R1[1], R2[1], R1R2_labels[1], n_perm, epoch_len, pre_bline, fs4pha, fs4pow, t_start, t_end, global_labels, s_pre)

                # corrolate power
                if not args.analysis or args.analysis == 'PC':
                    print('\ncomputing power correlations...')
                    epoch_len = 2
                    pre_bline = 0.1
                    freqs = np.arange(1,121)
                    n_dict = dict([(i, 3+(i-1)*9/(len(range(1,121))-1)) for i in range(1,121)]) # scale n with freq

                    R1_rw = R1[0]
                    R1_unrw = R1[1]

                    R2_rw = R2[0]
                    R2_unrw = R2[1]

                    R1_rw_norm = norm_power_bytrial(R1_rw, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)
                    R2_rw_norm = norm_power_bytrial(R2_rw, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)
                    rw_corr = corr_pwr(R1_rw_norm, R2_rw_norm)
                    np.save(f'{args.global_labels}_rw_pwr_corr.npy', rw_corr, allow_pickle=True)

                    R1_unrw_norm = norm_power_bytrial(R1_unrw, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)
                    R2_unrw_norm = norm_power_bytrial(R2_unrw, freqs, n_dict, args.s_pre, args.s_post, epoch_len, pre_bline)
                    unrw_corr = corr_pwr(R1_unrw_norm, R2_unrw_norm)
                    np.save(f'{args.global_labels}_unrw_pwr_corr.npy', unrw_corr, allow_pickle=True)

                    rw_corr_t = np.array(rw_corr).T
                    rw_corr_means = []
                    rw_corr_sems = []
                    for hz_corr in rw_corr_t:
                        mean = np.mean(hz_corr)
                        stderr = stats.sem(hz_corr)
                        rw_corr_means.append(mean)
                        rw_corr_sems.append(stderr)

                    lower_bound = np.array(rw_corr_means) + (-1*np.array(rw_corr_sems))
                    upper_bound = np.array(rw_corr_means) + np.array(rw_corr_sems)
                    plt.plot(freqs, rw_corr_means, color='darkgray')
                    plt.fill_between(freqs, lower_bound, upper_bound, color='darkgray', alpha=0.2)
                    #plt.show()

                    # corr_pow = (corrolate_power(R1_trial, R2_trial) for R1_trial, R2_trial in zip(R1_norm, R2_norm))
                    #
                    # print('plotting power correlations...\n')
                    # for cpow, label in zip(corr_pow, R1R2_labels):
                    #     plot_corrpow(cpow, epoch_len, pre_bline, freqs, args.global_labels, label)

                    # # filtered signal traces, fn calls and plot
                    # epoch_len = 1
                    # pre_bline = 0.1
                    # lc = 35
                    # hc = 45
                    # fs = 2000
                    # order = 4
                    #
                    # R1_filtered_sigs = (bp_filter_sigs(sig, lc, hc, fs, order, args.s_pre, args.s_post, epoch_len, pre_bline) for sig in R1)
                    # R2_filtered_sigs = (bp_filter_sigs(sig, lc, hc, fs, order, args.s_pre, args.s_post, epoch_len, pre_bline) for sig in R2)
                    #
                    # for R1_bp_sigs, R2_bp_sigs, label in zip(R1_filtered_sigs, R2_filtered_sigs, R1R2_labels):
                    #     plot_filtered_sigs_2R(R1_bp_sigs, R2_bp_sigs, lc, hc, epoch_len, pre_bline, args.global_labels, label)

                # Granger Causality
                if not args.analysis or args.analysis == 'Granger':
                    print('computing granger causality estimates...\n')
                    t_win = 100 # ms
                    order = 7   # integer
                    # in the container as (R2toR1, R1toR2 )
                    GC_estimates = (granger(R1_trials, R2_trials, t_win, order) for R1_trials, R2_trials in zip(R1, R2))

                    for i, (R2toR1, R1toR2, Ex_t, Ey_t, E_00, E_11) in enumerate(GC_estimates):
                        R2toR1 = np.squeeze(R2toR1)
                        R1toR2 = np.squeeze(R1toR2)
                        label = R1R2_labels[i]

                        print('now saving estimates... ')
                        np.save(args.global_labels + ' R2toR1_GCestimates ' + label + '.npy', R2toR1, allow_pickle=True)
                        np.save(args.global_labels + ' R1toR2_GCestimates ' + label + '.npy', R1toR2, allow_pickle=True)

                        # also saving raw estimates just in case
                        np.save(args.global_labels + ' Ex_t ' + label + '.npy', Ex_t, allow_pickle=True)
                        np.save(args.global_labels + ' Ey_t ' + label + '.npy', Ey_t, allow_pickle=True)
                        np.save(args.global_labels + ' E_00 ' + label + '.npy', E_00, allow_pickle=True)
                        np.save(args.global_labels + ' E_11 ' + label + '.npy', E_11, allow_pickle=True)

                    print('now calculating BIC...')
                    BIC = (granger(R1_trials, R2_trials, t_win, order, _bic=True) for R1_trials, R2_trials in zip(R1, R2))

                    for i, (bic) in enumerate(BIC):
                        label = R1R2_labels[i]
                        np.save(args.global_labels + ' BIC' + label + '.npy', bic, allow_pickle=True)

                # Granger Causality by trial
                if not args.analysis or args.analysis == 'Granger_bytrial':
                    print('computing granger causality estimates...\n')
                    t_win = 400 # ms
                    order = 6   # integer
                    # in the container as (R2toR1, R1toR2 )
                    GC_estimates = (granger_bytrial(R1_trials, R2_trials, t_win, order) for R1_trials, R2_trials in zip(R1, R2))

                    for i, (R2toR1, R1toR2) in enumerate(GC_estimates):
                        R2toR1 = np.squeeze(R2toR1)
                        R1toR2 = np.squeeze(R1toR2)
                        label = R1R2_labels[i]

                        print(f'shape: {np.shape(R1toR2)}')

                        print('now saving estimates... ')
                        np.save(args.global_labels + ' R2toR1_GCestimates_byTrial ' + label + '.npy', R2toR1, allow_pickle=True)
                        np.save(args.global_labels + ' R1toR2_GCestimates_byTrial ' + label + '.npy', R1toR2, allow_pickle=True)

                # Z-score Granger Causality
                if not args.analysis or args.analysis == 'GCez':
                    print('computing null GC estimates to z score...\n')
                    t_win = 200 # ms
                    order = 7
                    n_perm = 5000

                    start = datetime.datetime.now()
                    print(f'permutation testing began at {start}\n')

                    for i, (R1_trials, R2_trials) in enumerate(zip(R1, R2)):
                        zscored_GCe_y2x, zscored_GCe_x2y = get_zscored_GCe(R1_trials, R2_trials, t_win, order, n_perm)
                        label = R1R2_labels[i]

                        print('now saving z=scored GC estimates...\n')
                        np.save(args.global_labels + ' R2toR1_GCez ' + label + '.npy', zscored_GCe_y2x, allow_pickle=True)
                        np.save(args.global_labels + ' R1toR2_GCez ' + label + '.npy', zscored_GCe_x2y, allow_pickle=True)

                    end = datetime.datetime.now()
                    print(f'permutation testing ended at {end}')
                    print(f'duration of permutation testing: {end-start}')


                if not args.analysis or args.analysis == 'ppcorr':

                    epoch_len = 2
                    pre_bline = 0.1
                    freqs = np.arange(4,60,2)
                    n_dict = dict([(i, 3+(i-1)*9/(len(range(1,121))-1)) for i in range(1,121)]) # scale n with freq

                    for i in range(len(R1)):
                        R1_trials = R1[i]
                        R2_trials = R2[i]
                        local_label = R1R2_labels[i]
                        ppcorr_test = ppcorr(R1_trials, R2_trials, freqs, n_dict, epoch_len, pre_bline)
                        plot_ppcorr(ppcorr_test, freqs, args.global_labels, local_label, args.regions)

                if not args.analysis or args.analysis == 'lag_xcorr_bytrial':
                    print(R1R2_labels)

                    lc=4
                    hc=12
                    order=4

                    for i in range(len(R1)):
                        R1_trials = R1[i]
                        R2_trials = R2[i]
                        local_label = R1R2_labels[i]

                        xcorr = lag_xcorr_bytrial(R1_trials, R2_trials, lc, hc, order)
                        np.save(f'{args.global_labels} {local_label} Theta xcorr bytrial.npy', xcorr, allow_pickle=True)

                        plot_xcorr_bytrial(xcorr, f'{args.global_labels} {local_label}\nTheta amplitude cross correlation')


                print('analysis complete!\n')






if __name__ == '__main__':
    main()
