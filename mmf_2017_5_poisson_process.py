import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

def poisson_process(intensity):

    ## we are sampling the first sz jumps
    sz = 100

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    #######
    ### transform the uniform sample to exponentials
    #######

    sample = range(sz)
    for j in range(sz):
        sample[j] = dist.exponential_inverse_cdf(intensity,uni_sample[j])

    jumps = range(sz)
    for j in range(sz):
        if (j == 0):
            jumps[j] = sample[j]
        else:
            jumps[j] = jumps[j-1] + sample[j]

    ####
    ## plot the trajectory of the process
    ####

    steps = 1000
    step_size = 0.05
    x = [0] * steps
    y = [0] * steps

    for k in range(steps):
        x[k] = k * step_size
        for l in range(sz):
            if (jumps[l] > x[k]):
                y[k] = l
                break


    #######
    ### prepare and show plot
    ###
    plt.plot(x, y, 'r')
    plt.axis([0, steps * step_size, 0, max(y)*1.1])

    plt.xlabel('Time')
    plt.ylabel('# Of Jumps')
    plt.title("Histogram of Poisson Process with Intensity ={0}".format(intensity))

    plt.show()

if __name__ == '__main__':

    intensity = 0.25
    poisson_process(intensity)


