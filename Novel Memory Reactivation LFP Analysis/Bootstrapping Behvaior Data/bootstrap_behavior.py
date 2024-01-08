'''
Jonathan Ramos
11/21/2023
Small script to bootstrap behvaior data. We want to know if the subset of animals
either veh or abc that recieved electrodes reinstated differently from the population
'''
import numpy as np
import random
import math

def bootstrap(population, subset, n_iter):
    '''
    takes a population and one of its subsets and performs a permutation test
    to determine if the mean of the observed subset occured just by chance
    args:
        population: list of ints, the population observations
        subset: list of ints, a subset of population
        n_iter: int, the number of random samples to draw from population
            to create null distribution
    ret:
        p: float, p-value; proportion of observations in the null distribution
            that are at least as extreme as the observed population
        z: float, z-value; standardized difference of the means (the difference
            between the null distribution mean and the observed subset mean)
    '''
    # generate null distribution of means
    n_subset = len(subset)
    null_samples = np.array([random.sample(population, n_subset) for i in range(n_iter)])
    null_means = null_samples.mean(axis=1)
    null_medians = np.median(null_samples, axis=1)

    # compare observed mean to null distribution
    subset_mean = np.mean(subset)
    subset_median = np.median(subset)

    p_mean = len(null_means[null_means >= subset_mean]) / n_iter
    p_median = len(null_medians[null_medians >= subset_median]) / n_iter

    return p_mean, p_median

def permutation_test(sample, labels, n_iter):
    '''
    takes the observation from a sample containing two groups and performs a
    permutation test of the null hypotheses that the difference in mean or
    the difference in medians is 0 between the two groups. That is, under the
    null, we may freely exchange observations between groups because both groups
    come from the same distribution. If we reject the null, this means that the
    assignment of each observation into one group or ther other is not random.
    args:
        sample: list of ints, the sample observations
        labels: list of str, list of labels in the same order as sample
        n_iter: int, the number of samples to draw from the sample
            to create the null distribution
    ret:
        p: float, p-value; proportion of observations in the null distribution
            that are at least as extreme as the observed sample
    '''
    # toss into array for convenient indexing
    sample_arr = np.array(sample)
    labels_arr = np.array(labels)
    labels_unique = np.unique(labels_arr)

    # this fn is only valid for data with exactly two groups
    assert len(sample_arr) == len(labels_arr)
    assert len(labels_unique) == 2

    # compute test statistics
    group1_mean = np.mean(sample_arr[labels_arr == labels_unique[0]])
    group2_mean = np.mean(sample_arr[labels_arr == labels_unique[1]])
    test_mean_difference = abs(group1_mean - group2_mean)

    group1_median = np.median(sample_arr[labels_arr == labels_unique[0]])
    group2_median = np.median(sample_arr[labels_arr == labels_unique[1]])
    test_median_difference = abs(group1_median - group2_median)

    # generate permuted test statistics
    n_sample = len(sample)
    null_means = []
    null_medians = []

    for i in range(n_iter):
        # permute sample list, toss into array for convenient indexing
        permuted_sample = np.array(random.sample(sample, n_sample))

        # recompute test statistics under null hypothesis
        null_group1_mean = np.mean(permuted_sample[labels_arr == labels_unique[0]])
        null_group2_mean = np.mean(permuted_sample[labels_arr == labels_unique[1]])
        null_mean_difference = abs(null_group1_mean - null_group2_mean)

        null_group1_median = np.median(permuted_sample[labels_arr == labels_unique[0]])
        null_group2_median = np.median(permuted_sample[labels_arr == labels_unique[1]])
        null_median_difference = abs(null_group1_median - null_group2_median)

        # toss null test statistics into list
        null_means.append(null_mean_difference)
        null_medians.append(null_median_difference)

    # compute p values for null hypotheses
    null_means = np.array(null_means)
    null_medians = np.array(null_medians)

    p_mean = len(null_means[null_means >= test_mean_difference]) / n_iter
    p_median = len(null_medians[null_medians >= test_median_difference]) / n_iter

    return p_mean, p_median


def main():
    # from the prism we have the following active lever press data:
    veh_active = [62,10,80,7,65,24,43,47,24,9,30,17,43,17,22]
    abc_active = [5,17,1,16,11,5,18,1,5,28]
    population_active = veh_active + abc_active

    # from the prism, the following were marked as ephys
    veh_ephys_active = [30,17,43,17,22]
    abc_ephys_active = [1,5,28]

    # params
    n_iter = 10000
    seed = 1234567

    print('\nBootstrapped results (two-tailed p)')
    print(f'n of all possible combinations for Veh: 25 choose 5 = {math.comb(25,5)}')
    print(f'n of all possible combinations for ABC: 25 choose 3 = {math.comb(25,3)}')
    print(f'n_iter: {n_iter}')
    print(f'random seed: {seed}')

    random.seed(seed)
    p_mean, p_median = bootstrap(population_active, veh_ephys_active, n_iter)
    print(f'veh active:\n\tH0_mean: p={p_mean}\n\tH0_median p={p_median}')

    p_mean, p_median = bootstrap(population_active, abc_ephys_active, n_iter)
    print(f'abc active:\n\tH0_mean: p={1-p_mean}\n\tH0_median p={1-p_median}')

    ### repeat for rewarded data
    veh_rew = [29,6,42,5,33,19,25,20,11,6,20,13,9,15,6]
    abc_rew = [5,14,1,11,5,4,9,5,12,1]
    population_rew = veh_rew + abc_rew

    veh_ephys_rew = [20,13,9,15,6]
    abc_ephys_rew = [5,12,1]

    # params
    n_iter = 10000
    seed = 1234567

    print('\nBootstrapped results (two-tailed p)')
    print(f'n of all possible combinations for Veh: 25 choose 5 = {math.comb(25,5)}')
    print(f'n of all possible combinations for ABC: 25 choose 3 = {math.comb(25,3)}')
    print(f'n_iter: {n_iter}')
    print(f'random seed: {seed}')

    random.seed(seed)
    p_mean, p_median = bootstrap(population_rew, veh_ephys_rew, n_iter)
    print(f'veh reward:\n\tH0_mean: p={p_mean}\n\tH0_median p={p_median}')

    p_mean, p_median = bootstrap(population_rew, abc_ephys_rew, n_iter)
    print(f'abc reward:\n\tH0_mean: p={1-p_mean}\n\tH0_median p={1-p_median}')



    # ============= slightly different approach: permutation test ============= #

    # from the prism we have the following active lever press and label data:
    # 'b' label denotes animal was in behavior group,
    # 'e' label denotes animal was in ephys group
    veh_active = [62,10,80,7,65,24,43,47,24,9,30,17,43,17,22]
    veh_labels = ['b','b','b','b','b','b','b','b','b','b','b','b','e','e','e']
    seed = 1234567
    n_iter = 10000
    p_mean, p_median = permutation_test(veh_active, veh_labels, n_iter)

    print('\nPermutation test results (one-tailed p)')
    print(f'n_iter: {n_iter}')
    print(f'random seed: {seed}')
    print(f'veh active:\n\tH0_mean: p={p_mean}\n\tH0_median: p={p_median}')

    abc_active = [5,17,1,16,11,5,18,1,5,28]
    abc_labels = ['b','b','b','b','b','b','b','e','e','e']
    p_mean, p_median = permutation_test(abc_active, abc_labels, n_iter)
    print(f'abc active:\n\tH0_mean: p={p_mean}\n\tH0_median: p={p_median}')


    ### repeat for rewarded data
    veh_rew = [29,6,42,5,33,19,25,20,11,6,20,13,9,15,6]
    veh_labels = ['b','b','b','b','b','b','b','b','b','b','b','b','e','e','e']
    seed = 1234567
    n_iter = 10000
    p_mean, p_median = permutation_test(veh_rew, veh_labels, n_iter)

    print('\nPermutation test results (one-tailed p)')
    print(f'n_iter: {n_iter}')
    print(f'random seed: {seed}')
    print(f'veh reward:\n\tH0_mean: p={p_mean}\n\tH0_median: p={p_median}')

    abc_rew = [5,14,1,11,5,4,9,5,12,1]
    abc_labels = ['b','b','b','b','b','b','b','e','e','e']
    p_mean, p_median = permutation_test(abc_rew, abc_labels, n_iter)
    print(f'abc reward:\n\tH0_mean: p={p_mean}\n\tH0_median: p={p_median}')






if __name__ == '__main__':
    main()
