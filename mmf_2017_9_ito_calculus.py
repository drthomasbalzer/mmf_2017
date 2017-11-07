import numpy as np
import matplotlib.pyplot as plt

import mmf_2017_math_utilities as dist


### -- helper function - should be centralises
def random_sample_normal(sz, _timestep):

    sample = np.random.normal(0, np.sqrt(_timestep), sz)

    return sample

########
## -- Generic Base Class of C^{1,2} functions with derivatives where Ito's formula applies
########

class functionC12:

    def value(self, _t, _x):
        return 0.

    def f_t(self, _t, _x):
        return 0.

    def f_x(self, _t, _x):
        return 0.

    def f_xx(self, _t, _x):
        return 0.

########
## -- F(t,x) = x
########


class identityFunction(functionC12):

    def value(self, _t, _x):
        return _x

    def f_t(self, _t, _x):
        return 0.

    def f_x(self, _t, _x):
        return 1.

    def f_xx(self, _t, _x):
        return 0.

########
## -- F(t,x) = x^2
########

class squareFunction(functionC12):

    def value(self, _t, _x):
        return _x * _x

    def f_t(self, _t, _x):
        return 0.

    def f_x(self, _t, _x):
        return 2 * _x

    def f_xx(self, _t, _x):
        return 2.

########
## -- F(t,x) = \exp(x)
########

class exponentialFunction(functionC12):

    def value(self, _t, _x):
        return np.exp(_x)

    def f_t(self, _t, _x):
        return 0.

    def f_x(self, _t, _x):
        return self.value(_t, _x)

    def f_xx(self, _t, _x):
        return self.value(_t, _x)

########
## -- F(t,x) = \ln(x)
########

class logarithmicFunction(functionC12):

    def value(self, _t, _x):
        return np.log(_x)

    def f_t(self, _t, _x):
        return 0.

    def f_x(self, _t, _x):
        return 1. / _x

    def f_xx(self, _t, _x):
        return - 1. / (_x * _x)


class itoProcess:

    def __init__(self, a, b, x):

        ## standard Ito representation of a process
        ## X(t) = X(0) + \int_0^t a(u, X(u)) du + \int_0^t b(u, X(u)) dB(u)

        self.drift = a
        self.diffusion = b
        self.initial_value = x

    def a(self, _t, _x):
        return 0.

    def b(self, _t, _x):
        return 0.

class itoProcessExp(itoProcess):

    def a(self, _t, _x):
        return self.drift * _x

    def b(self, _t, _x):
        return self.diffusion * _x

class itoProcessStandard(itoProcess):

    def a(self, _t, _x):
        return self.drift


    def b(self, _t, _x):
        return self.diffusion


def plot_ito_process(_maxTime, _timestep, _number_paths, itoPr, funct):

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
        drivingProcess = 0
        path = [1] * (size + 1)
        for j in range(size + 1):
            if (j == 0):
                path[j] = funct.value(0, itoPr.initial_value)
                drivingProcess = itoPr.initial_value
                continue ## nothing
            else:

                ########
                ## the paths will be constructed through application of Ito's formula
                ########

                _x = drivingProcess
                _u = x[j]
                _du = _timestep
                _dBu = sample[i]
                _a = itoPr.a(_u, _x)
                _b = itoPr.b(_u, _x)

                ####
                ## -- path of actual F(t, X(t))
                ## F(t + dt, X(t + dt)) = F(t, X(t)) + F_t(t,X(t)) a(t,x) dt
                ##      + F_x(t, X(t)) a(t,x) dt + \frac{1}{2} F_xx(t,X(t)) b^2(t,x) dt
                ##      + F_x(t, X(t)) b(t,x) dB(t)
                ####

                path[j] = ( path[j - 1] + funct.f_t(_u, _x ) * _du
                    + _a * funct.f_x(_u, _x) * _du
                    + 0.5 * _b * _b * funct.f_xx(_u, _x) * _du
                    + _b * funct.f_x(_u, _x) * sample[i] )
                ####
                ## -- underlying ito process
                ## X(t + dt) = X(t) + a(t,x) dt + b(t,x) dB(t)
                ####

                drivingProcess = drivingProcess + _a * _du + _b * sample[i]

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
    plt.ylabel('Random Walk Value')
    plt.title("Paths of Ito Process")

    for k in range(_number_paths):
        plt.plot(x, paths[k])

    plt.show()


if __name__ == '__main__':

    max_time = 5
    timestep = 0.001
    paths = 10

    vol = 0.2

    f_id = identityFunction()
    f_exp = exponentialFunction()
    f_sq = squareFunction()
    f_ln = squareFunction()

    drift_0 = 0
    iPS = itoProcessStandard(drift_0, vol, 0.)

    drift_1 = 0
    iPE = itoProcessExp(drift_1, vol, 1.)

    plot_ito_process(max_time, timestep, paths, iPS, f_exp)
