import numpy as np
import matplotlib.pyplot as plt

### -- helper function - should be centralises
def random_sample_normal(sz, _timestep):
    sample = np.random.normal(0, np.sqrt(_timestep), sz)

    return sample


def terminal_utility_histogram(_b, _r, _sigma, T, _sample_size):
    #####
    ## plot the terminal utility of a stock vs an optimal strategy
    #####

    n = _sample_size  # this is how often we sample each time.

    sample = np.random.normal(0, T, n)

    alpha = .0
    sample_value_stock = range(n)
    sample_value_pi = range(n)
    sum_1 = 0.
    sum_2 = 0.
    pi = (_b - _r) / (_sigma * _sigma) / (1 - alpha)
    sigma_pi = _sigma * pi
    b_pi = _r + pi * (_b - _r)
    ## for
    for i in sample_value_stock:
        normal_sample = sample[i]
        stock_value = np.exp((_b + 0.5 * _sigma * _sigma) * T + _sigma * 1. * normal_sample)
        sample_portfolio = np.exp((b_pi + 0.5 * sigma_pi * sigma_pi) * T + sigma_pi * 1. * normal_sample)
        sum_1 = sum_1 + np.log(stock_value)
        sum_2 = sum_2 + np.log(sample_portfolio)
        sample_value_stock[i] = stock_value
        sample_value_pi[i] = sample_portfolio

    ##
    ## we then turn the outcome into a histogram
    ##
    num_bins = 100

    sum_1 = sum_1 / n
    sum_2 = sum_2 / n

    # the histogram of the data
    _n, bins, _hist = plt.hist(sample_value_stock, num_bins, normed=True, facecolor='green', alpha=0.75)
    plt.hist(sample_value_pi, num_bins, normed=True, facecolor='#727fff', alpha=0.75)
    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Terminal Wealth for Stock and Mixed Portfolio for $\pi=${0}".format(pi))

    plt.show()


if __name__ == '__main__':
    _b = .2
    _sigma = 0.5
    _r = 0.05
    _t = 1.
    _n = 75000

    terminal_utility_histogram(_b, _r, _sigma, _t, _n)

