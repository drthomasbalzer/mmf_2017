import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

###########
##### Demo of Moment Matching
###########

###########
##### Standard Sample First
###########

def moment_matching(p, sz_basket):

    #####
    ## create a basket with random weights of size sz_basket
    #####

    lower_bound = 0.
    upper_bound = 1.

    weights = np.random.uniform(lower_bound, upper_bound, sz_basket)

    ## calculate mean and variance of the basket

    expectation = 0
    variance = 0

    for k in range(sz_basket):
        expectation = expectation + weights[k]
        variance = variance + weights[k] * weights[k]

    expectation = expectation * dist.symmetric_binomial_expectation(p)
    variance = variance * dist.symmetric_binomial_variance(p)

    simulation = 50000

    outcome = range(simulation)

    for k in range(simulation):

        outcome[k] = 0

        uni_sample = np.random.uniform(lower_bound, upper_bound, sz_basket)

        #######
        ### transform the uniform sample
        #######
        sample = range(sz_basket)
        for j in range(sz_basket):
            sample[j] = dist.symmetric_binomial_inverse_cdf(p, uni_sample[j])

        for m in range(sz_basket):
            outcome[k] = outcome[k] + weights[m] * sample[m]

    num_bins = 50

    # the histogram of the data
    n, bins, _hist = plt.hist(outcome, num_bins, normed=True, facecolor='blue', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Basket Sample of Size={0}".format(sz_basket))

    y = range(0,num_bins+1)
    ###### overlay the moment matched pdf
    for i in range(0,num_bins+1):
         y[i] = dist.normal_pdf(bins[i], expectation, variance)

    plt.plot(bins, y, 'r*')
    plt.show()


if __name__ == '__main__':

    moment_matching(0.5, 20)

