import os
import re
import sys
import glob
import argparse

parser = argparse.ArgumentParser(description='renames files such that fname denotes the infusion recieved according to the training day; this is necessary for aggregate.py')
parser.add_argument('path', metavar='<path>', help='path of dir containing epoched signal data')
args = parser.parse_args()

files = glob.glob(args.path + '/*.npy')
d_drug = {
    'T1': 'coke',
    'T2': 'saline',
    'T3': 'coke',
    'T4': 'saline',
    'T5': 'coke',
    'T6': 'saline',
    'T7': 'coke',
    'T8': 'saline'
}

for f in files:
    day_regex = re.compile(r'T\d')
    day = day_regex.search(f.split('/')[-1]).group(0)
    f_new = f.replace('.npy', f'_{d_drug[day]}.npy')
    os.rename(f, f_new)
