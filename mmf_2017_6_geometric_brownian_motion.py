import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist

def random_sample_normal(sz):

    sample = np.random.normal(0, 1, sz)

    return sample


def geometric_brownian_motion(_time, _timestep, _number_paths, _volatility):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    size = int(_time / _timestep)

    total_sz = size * _number_paths

    sample = random_sample_normal(total_sz)

    paths = range(_number_paths)
    max_paths = range(_number_paths)

    mx = 0
    mn = 0

    x = [0.0] * (size + 1)

    for k in range(size + 1):
        x[k] = _timestep * k

    sq_t = np.sqrt(_timestep)

    v_t = _timestep * _volatility * _volatility

    ####
    ## plot the trajectory of the process
    ####
    i = 0
    for k in range(_number_paths):
        path = [1] * (size + 1)
        max_path = [1] * (size + 1)
        for j in range(size + 1):
            if (j == 0):
                continue ## nothing
            else:
                path[j] = path[j - 1] * np.exp(_volatility * sq_t * sample[i] - 0.5 * v_t)
                max_path[j] = max(max_path[j - 1], path[j])
                i = i + 1

        paths[k] = path
        max_paths[k] = max_path

        mx = max(mx, max(path))
        mn = min(mn, min(path))

    #######
    ### prepare and show plot
    ###
    plt.axis([0, max(x), 1.1 * mn, 1.1 * mx])
    plt.xlabel('Time')
    plt.ylabel('Random Walk Value')
    plt.title("Paths of Geometric Brownian Motion with Volatility={0}".format(_volatility))

#    plt.plot(x, max_paths[0])
#    plt.plot(x, paths[0])

    plot_max = False

    if (plot_max):
        for k in range(_number_paths):
            plt.plot(x, max_paths[k])
    else:
        for k in range(_number_paths):
            plt.plot(x, paths[k])


    plt.show()


if __name__ == '__main__':

    time = 5
    timestep = 0.001
    paths = 15
    volatility = 0.2
    geometric_brownian_motion(time, timestep, paths, volatility)


