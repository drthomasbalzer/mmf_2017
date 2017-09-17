import numpy as np
import math
import matplotlib.pyplot as plt

###########
##### Demo of Quantile Transformation
###########


###########
##### Standard Sample First
###########


def uniform_histogram(sz):

    lower_bound = 0.
    upper_bound = 1.

    sample = np.random.uniform(lower_bound, upper_bound, sz)

    num_bins = 50

    # the histogram of the data
    plt.hist(sample, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Uniform Sample of Size={0}".format(sz))

    plt.show()

######
## pdf and inverse distribution function of $Exp(\lambda)$-distribution
######

def pdf_exponential(_lambda, x):

    return _lambda * np.exp(-_lambda * x)


def inverse_cdf_exponential(_lambda, x):

    return -1./_lambda * np.log(1-x)

######
## inverse distribution function of $B(1,p)$-distribution
######

def inverse_cdf_binomial(p, x):

    if (x < 1-p):
        return 0.
    else:
        return 1.

#####
## Create distribution via Quantile Transform -- $B(1,p)$-distribution
#####

def binomial_histogram(p, sz):

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    #######
    ### transform the uniform sample
    #######
    sample = range(sz)
    for j in range(sz):
        sample[j] = inverse_cdf_binomial(p,uni_sample[j])

    num_bins = 100

    # the histogram of the data
    plt.hist(sample, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Binomial Sample with Success Probability={0}".format(p))

    plt.show()

def exponential_histogram(_lambda, sz):

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    #######
    ### transform the uniform sample
    #######
    sample = range(sz)
    for j in range(sz):
        sample[j] = inverse_cdf_exponential(_lambda,uni_sample[j])

    num_bins = 50

    # the histogram of the data
    n, bins, _hist = plt.hist(sample, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Exponential Sample with Parameter={0}".format(_lambda))

    y = range(0,num_bins+1)
    ###### overlay the actual pdf
    for i in range(0,num_bins+1):
         y[i] = pdf_exponential(_lambda, bins[i])

    plt.plot(bins, y, 'r--')
    # # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


if __name__ == '__main__':

    calc_type = 2

    size = 10000

    if (calc_type == 0):  ### uniform sample
        uniform_histogram(size)
    else:
        if (calc_type == 1): ### generate a binomial distribution
            p = 0.40
            binomial_histogram(p, size)
        else: ### generate an exponential distribution
            _lambda = 1.
            exponential_histogram(_lambda, size)

