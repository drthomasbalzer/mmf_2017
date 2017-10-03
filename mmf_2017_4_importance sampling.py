import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist


def exponential_importance_sampling(_lambda, shift, strike, digitalPayout=True):
    ## we evaluate a payout of $P(X > K)$ for an exponential distribution with a given _lambda and a shifted model where
    ## the shift is set to bring the mean of the exponentially tilted distribution to the strike that is considered

    repeats = 500

    sample_is = [0] * repeats
    sample_non_is = [0] * repeats

    for z in range(repeats):

        ## we are sampling sz times in each iteration

        sz = 5000

        lower_bound = 0.
        upper_bound = 1.

        ## we create a uniform sample first;

        uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

        #######
        ### transform the uniform sample to exponentials
        #######

        sample_exp = range(sz)
        for j in range(sz):
            sample_exp[j] = dist.exponential_inverse_cdf(_lambda, uni_sample[j])

        sample_exp_shift = range(sz)
        for j in range(sz):
            sample_exp_shift[j] = dist.exponential_inverse_cdf(_lambda + shift, uni_sample[j])

        ### evaluate the payout

        payout_non_is = 0
        payout_is = 0

        c = (_lambda + shift) / _lambda

        for k in range(sz):
            if (sample_exp[k] > strike):
                if (digitalPayout):
                    payout_non_is = payout_non_is + 1.
                else:
                    payout_non_is = payout_non_is + (sample_exp[k] - strike)

            if (sample_exp_shift[k] > strike):
                if (digitalPayout):
                    payout_is = payout_is + np.exp(shift * sample_exp_shift[k]) / c
                else:
                    payout_is = payout_is + np.exp(shift * sample_exp_shift[k]) / c * (sample_exp_shift[k] - strike)

        payout_non_is = payout_non_is / sz
        payout_is = payout_is / sz

        sample_non_is[z] = payout_non_is
        sample_is[z] = payout_is

    #######
    ### prepare and show plot
    ###
    num_bins = 50

    ### this is the exact result
    p = np.exp(-strike * _lambda)

    # the histogram of the data
    plt.hist(sample_non_is, num_bins, normed=True, facecolor='green', alpha=0.75)
    plt.hist(sample_is, num_bins, normed=True, facecolor='#d57307', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    if (digitalPayout):
        plt.title("Digital Payout (Value = {0}) with and w/o Importance Sampling.".format(p))
    else:
        plt.title("Option Payout with and w/o Importance Sampling.")
    plt.show()


if __name__ == '__main__':
    intensity = 0.275
    strike = 15
    shift = 1. / strike - intensity
    digitalPayout = True
    exponential_importance_sampling(intensity, shift, strike, digitalPayout)
