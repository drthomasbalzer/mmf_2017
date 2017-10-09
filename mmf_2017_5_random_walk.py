import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

def random_sample_sym_binomial(_p, sz):

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    sample = range(sz)
    for j in range(sz):
        sample[j] = dist.symmetric_binomial_inverse_cdf(_p,uni_sample[j])

    return sample


def random_walk(_p, _steps, _paths, scaling):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    scaled_steps = _steps * scaling
    sz = scaled_steps * _paths

    sample = random_sample_sym_binomial(_p, sz)

    paths = range(_paths)

    mx = 0
    mn = 0

    x = [0.0] * (scaled_steps + 1)

    for k in range(scaled_steps + 1):
        x[k] = float(k / float(scaling))


    ####
    ## plot the trajectory of the process
    ####
    i = 0
    for k in range(_paths):
        path = [0] * (scaled_steps + 1)
        for j in range(scaled_steps + 1):
            if (j == 0):
                path[j] = 0
            else:
                path[j] = path[j-1] + sample[i] / np.sqrt(scaling)
                i = i + 1



        paths[k] = path
        mx = max(mx, max(path))
        mn = min(mn, min(path))

    #######
    ### prepare and show plot
    ###
    plt.axis([0, max(x), 1.1 * mn, 1.1 * mx])
    plt.xlabel('Time')
    plt.ylabel('Random Walk Value')
    plt.title("Paths of Random Walk with Probability={0}".format(p))

    for k in range(_paths):
        plt.plot(x, paths[k])

    plt.show()

def random_walk_hist(_p, _steps, _paths, scaling):

    scaled_steps = _steps * scaling
    sz = scaled_steps * _paths

    sample = random_sample_sym_binomial(_p, sz)

    output = range(_paths)

    ####
    ## plot the trajectory of the process
    ####
    i = 0
    for k in range(_paths):
        path = 0
        for j in range(scaled_steps + 1):
            if (j == 0):
                path = 0
            else:
                path = path + sample[i] / np.sqrt(scaling)
                i = i + 1

        output[k] = path

    #######
    ### prepare and show plot
    ###

    num_bins = _steps / 1.2
    plt.hist(output, num_bins, normed=True, facecolor='blue', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')

    plt.show()

if __name__ == '__main__':

    p = 0.5
    _paths = 1000
    _steps = 50
    scaling = 1
#    random_walk(p, _steps, _paths, scaling)
    random_walk_hist(p, _steps, _paths, scaling)


