import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

def random_sample_normal(sz, _timestep):

   sample = np.random.normal(0, np.sqrt(_timestep), sz)

   return sample

def plot_bachelier_digital_option(_time, _timestep, _strike):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    size = int(_time / _timestep)

    sample = random_sample_normal(size, _timestep)

    sq_t = np.sqrt(_timestep)

    path_bm = [0] * (size)
    path_digital_option = [0] * (size)
    path_digital_option_hedge = [0] * (size)

    mx = 0
    mn = 0

    x = [0.0] * (size)

    for k in range(size):
        x[k] = _timestep * k


    ####
    ## plot the trajectory of the process
    ####
    for j in range(size):
        _t_remain = np.sqrt(_time - x[j])
        if (j == 0):
            path_bm[j] = 0.
            path_digital_option[j] = 1 - dist.standard_normal_cdf((_strike - path_bm[j]) / _t_remain)
            path_digital_option_hedge[j] = path_digital_option[j]
        else:
            path_bm[j] = path_bm[j - 1] + sample[j]
            path_digital_option_hedge[j] = path_digital_option_hedge[j-1] + sample[j] * dist.standard_normal_pdf((_strike - path_bm[j-1]) / _t_remain) / _t_remain
            path_digital_option[j] = 1 - dist.standard_normal_cdf((_strike - path_bm[j]) / _t_remain)

        mx = max(mx, max(path_digital_option))
        mn = min(mn, min(path_digital_option))

    #######
    ### prepare and show plot
    ###
    plt.axis([0, max(x), 1.1 * mn, 1.1 * mx])

    plt.title("Paths of Digital Option Value")

    plt.subplot(2, 1, 1)
    plt.plot(x, path_digital_option)
    plt.plot(x, path_digital_option_hedge)
    plt.ylabel('Option Value')

    plt.subplot(2, 1, 2)
    plt.plot(x, path_bm, 'r-')
    plt.ylabel('Underlying Value')

    plt.show()


if __name__ == '__main__':

    time = .5
    timestep = 0.0005
    strike = .25
    plot_bachelier_digital_option(time, timestep, strike)
