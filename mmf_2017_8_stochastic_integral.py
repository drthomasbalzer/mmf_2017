import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

def stochastic_integral_hist(_p, _steps, _paths, scaling):
    scaled_steps = _steps * scaling
    sz = scaled_steps * _paths

    output_lhs = range(_paths)
    output_rhs = range(_paths)
    output_lhs_non_si = range(_paths)
    output_rhs_non_si = range(_paths)

    delta_t = 1. / scaling
    delta_t_sq = 1. / np.sqrt(scaling)
    ####
    ## plot the trajectory of the process
    ####

    _lambda = 1

    for k in range(_paths):
        i = 0
        sample = np.random.normal(0., 1., scaled_steps + 1)
        bm_path = 0
        si_lhs = 0
        si_rhs = 0
        non_si_lhs = 0
        non_si_rhs = 0
        for j in range(scaled_steps + 1):
            if (j == 0):
                bm_path = 0
            else:
                increment = sample[i] * delta_t_sq
                ## evaluate the stochastic integral at the left hand side
                si_lhs = si_lhs + bm_path * increment
                non_si_lhs = non_si_lhs + bm_path * delta_t
                ## roll the path to the right hand side
                bm_old = bm_path
                bm_path = bm_path + increment
                si_rhs = si_rhs + (_lambda * bm_path + (1 - _lambda) * bm_old) * increment
                non_si_rhs = non_si_rhs + bm_path * delta_t
            i = i + 1

        output_lhs[k] = si_lhs
        output_rhs[k] = si_rhs
        output_lhs_non_si[k] = non_si_lhs
        output_rhs_non_si[k] = non_si_rhs

    num_bins = 50
    _n, bins, _hist = plt.hist(output_lhs, num_bins, normed=True, facecolor='blue', alpha=0.75)
    plt.hist(output_rhs, num_bins, normed=True, facecolor='#12ff26', alpha=0.75)

    show_non_si = True

    if show_non_si:
        plt.hist(output_rhs_non_si, num_bins, normed=True, facecolor='#0059ff', alpha=0.5)
        plt.hist(output_lhs_non_si, num_bins, normed=True, facecolor='#ff9772', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Stochastic Integral $\int_0^t B(u) dB(u)$ for 2 approximations")

    plt.show()


if __name__ == '__main__':
    p = 0.5
    _paths = 5000
    _steps = 1
    scaling = 500
    stochastic_integral_hist(p, _steps, _paths, scaling)
