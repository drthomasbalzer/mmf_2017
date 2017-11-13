import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist


### -- helper function - should be centralises
def random_sample_normal(sz, _timestep):

    sample = np.random.normal(0, np.sqrt(_timestep), sz)

    return sample

#############
##  -- Generic Representation of an Ito Process
##  --- of the form dX(t) = (A(t)X(t) + a(t)) dt + (D(t) X(t) + d(t)) dB(t)
#############

class itoProcessGeneral:

    def drift(self, _t, _x):
        return 0.

    def diffusion(self, _t, _x):
        return 0.


class itoProcessTH(itoProcessGeneral):

    def __init__(self, _A, _a, _D, _d, _x):

        self._A = _A
        self._a = _a
        self._D = _D
        self._d = _d

        self.initial_value = _x

    def drift(self, _t, _x):
        return self._A * _x + self._a

    def diffusion(self, _t, _x):
        return self._D * _x + self._d

class itoProcessBrownianBridge(itoProcessGeneral):

    def __init__(self, _T, _b, _sigma, _x):

        self._T = _T
        self._b = _b
        self._sigma = _sigma

        self.initial_value = _x

    def drift(self, _t, _x):
        return (self._b - _x) / (self._T - _t)

    def diffusion(self, _t, _x):
        return self._sigma



def plot_sde(_maxTime, _timestep, _number_paths, itoPr):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    ## normals for a single paths

    size = int(_maxTime / _timestep)

    ## total random numbers needed
    total_sz = size * _number_paths

    sample = random_sample_normal(total_sz, _timestep)

    paths = range(_number_paths)

    x = [0.0] * (size + 1)

    for k in range(size + 1):
        x[k] = _timestep * k

    mx = 0
    mn = 0

    ####
    ## plot the trajectory of the Ito process
    ####
    i = 0
    for k in range(_number_paths):
        process = 0
        path = [1] * (size + 1)
        for j in range(size + 1):
            if (j == 0):
                process = itoPr.initial_value
                path[j] = process
                continue ## nothing
            else:

                ########
                ## the paths will be constructed through application of Ito's formula
                ########

                _x = process
                _u = x[j-1]
                _du = _timestep
                _dBu = sample[i]
                _a = itoPr.drift(_u, _x)
                _b = itoPr.diffusion(_u, _x)

                ####
                ## -- underlying ito process
                ## X(t + dt) = X(t) + a(t,x) dt + b(t,x) dB(t)
                ####

                process = process + _a * _du + _b * sample[i]
                path[j] = process

                ### increment counter for samples

                i = i + 1

        paths[k] = path

        mx = max(mx, max(path))
        mn = min(mn, min(path))

    #######
    ### prepare and show plot
    ###
    plt.axis([0, max(x), 1.1 * mn, 1.1 * mx])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title("Paths of Ito Process")

    for k in range(_number_paths):
        plt.plot(x, paths[k])

    plt.show()


if __name__ == '__main__':

    max_time = 5
    timestep = 0.001
    paths = 6

    ### ito process of the form dX = X(a dt + b dB(t))
    ito_exp = itoProcessTH(0.2, 0, 0.2, 0., 1)

    ### ito process of the form dX = a dt + b dB(t)
    ito_bm = itoProcessTH(0., 0.2, 0., 0.2, 0)

    ### ito process of the form dX = X(t) dt + b dB(t)
    ito_1 = itoProcessTH(0, 0.0, 0., 0.5, 1)

    ### ito process of the form dX = mean X(t) dt + b dB(t)
    ito_2 = itoProcessTH(-0.05, 0.0, 0., 0.05, 1)

    ### ito process of the form dX = mean X(t) dt + b dB(t)
    ##
    ito_mr = itoProcessTH(-.1, .2, 0., 0.2, .5)

    ##
    ### Brownian Bridge
    ##
    ito_bb = itoProcessBrownianBridge(max_time, 1., .25, 0.)

    plot_sde(max_time, timestep, paths, ito_bb)
