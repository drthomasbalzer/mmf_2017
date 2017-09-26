import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

###########
##
## Demo of the Law of Large Numbers
##
###########

def binomial_lln(sample_size, p):

    ## we are sampling from a $B(1,p)$ distribution
    ##

    n = sample_size
    ######
    ## Step 1 - create sample of independent uniform random variables

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

    ######
    ## Step 2 - transform them to $B(1,p)$ distribution

    sample = range(n)
    for j in range(n):
        sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

    x_val = range(n)  # values on the x axis
    sum = 0
    prev_sum = 0
    y_val = range(n)  # y_values - running average
    y_average = [p for x in range(n)]  # y_values - actual average



    ######
    ## we want to plot the cumulative average of all the samples

    for k in range(n):
        x_val[k] = k
        if (k == 0):
            sum = sample[k]
        else:
            prev_sum = sum
            sum = sum + sample[k-1]

        y_val[k] = sum / (k+1)

    #######
    ### prepare and show plot
    ###
    plt.plot(x_val, y_val, 'r-')
    plt.plot(x_val, y_average, 'g-')
    plt.axis([min(x_val), max(x_val), p - 0.2, p+0.2])
    plt.title("Cumulative Average")
    plt.xlabel("x")
    plt.ylabel("Average")
    plt.show()
    ###
    #######

def binomial_lln_hist(sample_size, repeats, p):

    ##
    ## we are sampling from a $B(1,p)$ distribution
    ##

    n = sample_size  # this is how often we sample each time.

    lower_bound = 0.
    upper_bound = 1.

    plotCLN = True
    sample_value = range(repeats)

    for i in sample_value:

        prev_sum = 0
        sum = 0

        ## Step 1 - create sample of independent uniform random variables

        ### This code uses uniform random variables and the quantile transform
        uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

        sample = range(n)
        for j in range(n):
            sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

        for k in range(n):
            if (k == 0):
                sum = sample[k]
            else:
                prev_sum = sum
                sum = sum + sample[k - 1]


        if plotCLN:
            sample_value[i] = (sum - sample_size * p) / np.sqrt(sample_size * p * (1-p))
        else:
            sample_value[i] = sum / (sample_size)

            ## sample value for CLN


    ##
    ## we then turn the outcome into a histogram
    ##
    num_bins = 50

    # the histogram of the data
    _n, bins, _hist = plt.hist(sample_value, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Average Sample of Size={1} of B(1,p) with p={0}".format(p, sample_size))

    ###### overlay the actual normal distribution
    if plotCLN:

        y = range(0,num_bins+1)
        for m in range(0,num_bins+1):
             y[m] = dist.standard_normal_pdf(bins[m])

        plt.plot(bins, y, 'r--')

    plt.show()

    #######

if __name__ == '__main__':

    sz = 15000
    p = .75
#    binomial_lln(sz, p)

    repeats = 2000
    binomial_lln_hist(sz, repeats, p)


