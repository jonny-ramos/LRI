import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from lfp import load_data, erp

parser = argparse.ArgumentParser(description='loads in dict of trialzed recording data and computes and plots erp')
parser.add_argument('path', metavar='<path>', help='path of curated data to load')
args = parser.parse_args()

def main():
    label, data = load_data(args.path)
    fs = 2000

    # take data out of dict and toss into data matrix, np.ndarray(L, N): where
    # L: number of trials, N: number of samples
    data = np.array(list(data.values()))
    n = np.shape(data)[1]

    # compute erp
    arr = erp(data)

    # plot
    f, ax = plt.subplots(1,1)
    f.set_figheight(4)
    f.set_figwidth(7)

    t_zero = 2000
    time = np.arange((0 - t_zero) / fs, (n - t_zero) / fs, 1/fs)
    title = ' '.join(label.split('_'))

    ax.plot(time, arr)
    ax.set_title(title, size=16)
    ax.set_xlabel('Time (ms), press @ t=0', fontsize=16)
    ax.set_ylabel('potential (µV)', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    f.savefig(' '.join([title, 'ERP.png']), bbox_inches='tight', dpi=600)

    sys.exit()

if __name__ == '__main__':
    main()
