import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

def random_sample_normal(sz, _timestep):

   sample = np.random.normal(0, np.sqrt(_timestep), sz)

   return sample

def plot_maximising_goal_probability(_time, _timestep, _initial_capital, _target, _b, _r, _sigma):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    size = int(_time / _timestep) - 1

    sample = random_sample_normal(size, _timestep)

    sq_t = np.sqrt(_timestep)

    path_underlying = [0] * (size)
    path_wealth = [0] * (size)
    path_portfolio = [0] * (size)

    mx = 0
    mn = 0

    x = [0.0] * (size)

    for k in range(size):
        x[k] = _timestep * k

    _theta = (_b - _r) / _sigma

    _y0 = np.sqrt(_time) * dist.standard_normal_inverse_cdf(_initial_capital * np.exp(_r * _time) / _target)
    ####
    ## create the various paths for plotting
    ####
    bm = 0
    for j in range(size):
        _t_remain = _time - x[j]
        _t_sq_remain = np.sqrt(_t_remain)
        if (j == 0):
            path_underlying[j] = 1.
            path_wealth[j] = _initial_capital
        else:
            path_underlying[j] = path_underlying[j-1] * (1. + _b * _timestep + _sigma * sample[j])
            bm = bm + sample[j] + _theta * _timestep
            path_wealth[j] = _target * np.exp( - _r * _t_remain ) * \
                                dist.standard_normal_cdf((bm + _y0) / _t_sq_remain)

        _y = path_wealth[j] * np.exp(_r * _t_remain) / _target
        path_portfolio[j] = dist.standard_normal_pdf(dist.standard_normal_inverse_cdf(_y)) / (_y * _sigma * _t_sq_remain)

        mx = max(mx, max(path_wealth))
        mn = min(mn, min(path_wealth))

    #######
    ### prepare and show plot
    ###
    plt.axis([0, max(x), 1.1 * mn, 1.1 * mx])

    plt.subplot(3, 1, 1)
    plt.title("Maximising Probability of Reaching a Goal")

    plt.plot(x, path_underlying)
    plt.ylabel('Stock Price')

    plt.subplot(3, 1, 2)
    plt.plot(x, path_wealth, 'r-')
    plt.ylabel('Wealth Process')

    plt.subplot(3, 1, 3)
    plt.plot(x, path_portfolio, 'r-')
    plt.ylabel('Portfolio Value')

    plt.show()


if __name__ == '__main__':

    _initial_capital = 1
    _target_wealth = 1.20

    _time = 2.
    timestep = 0.001

    _b = 0.08
    _r = 0.05
    _sigma = .30

    plot_maximising_goal_probability(_time, timestep, _initial_capital, _target_wealth, _b, _r, _sigma)
