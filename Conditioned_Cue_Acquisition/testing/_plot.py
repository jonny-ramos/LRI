import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description = 'plot traces for visual curation')
parser.add_argument('PFCpath', metavar='<path>', help='path to dict of PFC data to curate.')
parser.add_argument('HPCpath', metavar='<path>', help='path to dict of HPC data to curate.')
args = parser.parse_args()

def _get_fig(d1, d2, key):
    PFC_trial = d1[key]
    HPC_trial = d2[key]

    fig = plt.figure(figsize=(6,3))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    fig.suptitle(key)
    axs[0].plot(PFC_trial)
    axs[1].plot(HPC_trial, c='orange')

    for ax in axs:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(f'temp/{key}.png')
    plt.close()

def main():
    PFC = np.load(args.PFCpath, allow_pickle=True).item()
    HPC = np.load(args.HPCpath, allow_pickle=True).item()

    assert PFC.keys() == HPC.keys()
    keys = list(PFC.keys())

    if os.path.exists('temp/') and os.path.isdir('temp/'):
        shutil.rmtree('temp/')
    os.mkdir('temp/')

    for k in keys:
        _get_fig(PFC, HPC, k)

if __name__ == '__main__':
    main()
