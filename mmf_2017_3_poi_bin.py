import matplotlib.pyplot as plt
import numpy as np

import mmf_2017_math_utilities as dist

def comparison_poi_binom(lam, upper_bound):

    check_sum = 0
    n = upper_bound + 1
    v_poi = range(n)
    v_binom = range(n)
    x_axis = range(n)
    for k in range(0, n):
        p_poi = dist.poisson_pdf(lam, k)
        p_binom = dist.binomial_pdf(lam / n, k, n)
        x_axis[k] = k
        v_poi[k] = p_poi
        v_binom[k] = p_binom
        check_sum = check_sum + p_poi

    plt.plot(x_axis, v_poi, 'ro')
    plt.plot(x_axis, v_binom, 'bo')
    plt.axis([-0.5, n, 0, max(v_poi) * 1.1])
    plt.title("Poisson Vs Binomial distribution for lambda={0}".format(lam))
    plt.xlabel("# Successes")
    plt.ylabel("Probability")
    plt.show()
    print check_sum

def poisson_plot(lam, upper_bound):

    check_sum = 0
    n = upper_bound + 1
    v_poi = range(n)
    x_axis = range(n)
    for k in range(0, n):
        p_poi = dist.poisson_pdf(lam, k)
        x_axis[k] = k
        v_poi[k] = p_poi

    plot_single_probability_graph("Poisson", lam, n, x_axis, v_poi)

def binomial_plot(p, n):

    check_sum = 0
    y_max = 0
    values = range(n)
    x_axis = range(n)
    for k in range(0, n):
        p_k = dist.binomial_pdf(p, k, n)
        x_axis[k] = k
        values[k] = p_k
        check_sum = check_sum + p_k

    plot_single_probability_graph("Binomial", p, n, x_axis, values)

    print check_sum

#####
## Create distribution via Quantile Transform -- $B(n,p)$- vs $Poi(\lambda)$-distribution
#####

def bin_vs_poisson_histogram(_lambda, n, sz):

    lower_bound = 0.
    upper_bound = 1.

    p = _lambda / n
    total_sample_size = n * sz
    print total_sample_size
    uni_sample = np.random.uniform(lower_bound, upper_bound, total_sample_size)

    #######
    ### transform the uniform sample
    #######
    sample = range(total_sample_size)
    for j in range(total_sample_size):
        sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

    outcome_bnp = [0]* sz
    index = 0
    for k in range(sz):
        for l in range(n):
            outcome_bnp[k] = outcome_bnp[k] + sample[index]
            index = index + 1

    num_bins = 30

    # the histogram of the data

    plt.hist(outcome_bnp, num_bins, normed=True, facecolor='green', alpha=0.75)
#    bnp_sample = np.random.binomial(n, p, sz)
#    plt.hist(bnp_sample, num_bins, normed=True, facecolor='red', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')

    plt.title("Histogram of B(n,p) sample with p={0} and n={1}".format(p, n))

    plt.show()

##########
#### helper function to plot graph of a probability distribution
##########

def plot_single_probability_graph(str_dist, param_1, param_2, x_axis, y_axis):

    plt.plot(x_axis, y_axis, 'ro-')
    plt.axis([-0.5, max(x_axis), 0, max(y_axis) * 1.1])
    plt.title(str_dist + " distribution for p={0} and n={1}".format(param_1, param_2))
    plt.xlabel("# Successes")
    plt.ylabel("Probability")
    plt.show()
    return

if __name__ == '__main__':

#    binomial_plot(0.2, 50)
#    poisson_plot(0.5, 10)
#   comparison_poi_binom(0.5,10)
    _lambda = 0.5
    n = 20
    sz = 10000
    bin_vs_poisson_histogram(_lambda, n, sz)


