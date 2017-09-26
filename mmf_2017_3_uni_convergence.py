import numpy as np
import matplotlib.pyplot as plt

def uniform_histogram_powers(sz, pow_1, pow_2):

    lower_bound = 0.
    upper_bound = 1.

    sample = np.random.uniform(lower_bound, upper_bound, sz)

    num_bins = 50

    sample_1 = range(sz)
    sample_2 = range(sz)

    for k in range(sz):
        sample_1[k] = np.power(sample[k], pow_1)
        sample_2[k] = np.power(sample[k], pow_2)

    # the histogram of the data
    plt.hist(sample_2, num_bins, normed=True, facecolor='blue', alpha=0.5)
    plt.hist(sample_1, num_bins, normed=True, facecolor='red', alpha=0.75)
    plt.hist(sample, num_bins, normed=True, facecolor='green', alpha=1)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Uniform Sample of Size={0}".format(sz))

    plt.show()

if __name__ == '__main__':

    sz = 250000
    pow_1 = 1
    pow_2 = 1.25
    uniform_histogram_powers(sz, pow_1, pow_2)


