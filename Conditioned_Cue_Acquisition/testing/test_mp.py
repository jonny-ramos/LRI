import numpy as np
import itertools
import matplotlib.pyplot as plt
import multiprocessing
import sys
import os

# let's just try some simple math first as an example
def prod(a, b):
    return a * b

def wrapper(fpair):
    fp, fa = fpair
    p = ((fp, fa), prod(fp, fa))

    return p

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 4)
    f_pha_low, f_pha_high = tuple([int(n) for n in '4,25'.split(',')]) # increased to 25 from 12 for visualization
    f_amp_low, f_amp_high = tuple([int(n) for n in '20,100'.split(',')])
    freqs4pha = np.linspace(f_pha_low, f_pha_high, (f_pha_high - f_pha_low)*2 + 1)
    freqs4amp = np.linspace(f_amp_low, f_amp_high, (f_amp_high - f_amp_low) + 1)


    n_pha = len(freqs4pha)
    n_amp = len(freqs4amp)


    # ### test partitioning
    # partial1 = wrapper(freqs4pha[:int(n_pha//2)], freqs4amp[:int(n_amp//2)])
    # partial2 = wrapper(freqs4pha[int(n_pha//2):], freqs4amp[int(n_amp//2):])
    # partial3 = wrapper(freqs4pha[:int(n_pha//2)], freqs4amp[int(n_amp//2):])
    # partial4 = wrapper(freqs4pha[int(n_pha//2):], freqs4amp[:int(n_amp//2)])

    fpairs = list(itertools.product(freqs4pha, freqs4amp))
    pool = multiprocessing.Pool(processes=2)
    result = pool.map(wrapper, fpairs)

    # sort based on (f,f) pair
    result.sort(key=lambda x: x[0])

    # access only product from result
    val = list(map(lambda x: x[1], result))

    # reshape for plotting
    cm = np.reshape(val, (n_pha, n_amp)).T

    # plot
    plt.imshow(cm)
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()

    sys.exit()
