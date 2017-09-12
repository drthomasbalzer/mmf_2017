import numpy as np
import math
import matplotlib.pyplot as plt

###############
##
##  plot a continuously compounding account against one with "randdomness"
##
###############

def compounding_plot(rate, perturbationRate, show):

    #############
    #### simple comparison of a "risk-free" vs a "risky" account
    #############

    t_min = 0
    t_max = 10.0
    step = 0.05
    n = int((t_max - t_min) / step)
    x_val = range(n) # values on the x axis
    y_val = range(n) # this is the analytic solution of the ODE-- just the exponential
    y_val_sc = range(n) # compounding term with discrete (small time step compounding)
    y_val_nc = range(n) # no compounding - simple interest to maturity
    y_val_sc_pt = range(n)  # compounding term with discrete compounding and perturbation
 #   y_val_gbm = range(n)  # analytic solution of SDE

    starting_value = 1.
    perturb = False
    if (perturbationRate <> 0):
        perturb = True
        vol = perturbationRate

    for k in range(0, n):
        x_val[k] = t_min + k * step;
        y_val[k] = np.exp(rate * x_val[k])
        p_rate = rate * step
        random_shock = np.random.normal(0, vol * np.sqrt(step), 1)
        if perturb:
            ### perturb the growth rate with a normally distributed (scaled for $\sqrt{timestep}$ volatility$)
            p_rate = p_rate + random_shock
        if (k == 0): # initial value
            y_val_sc[k] = starting_value
            y_val_sc_pt[k] = starting_value
#            y_val_gbm[k] = starting_value
        else:
            y_val_sc[k] = y_val_sc[k-1] * (1 + rate * step)  ## compounding at a "random rate"
            y_val_sc_pt[k] = y_val_sc_pt[k-1] * (1 + p_rate)  ## compounding at a "random rate"
#            y_val_gbm[k] = y_val_gbm[k-1] * np.exp(rate * step + random_shock - 0.5 * vol * vol * step)

        y_val_nc[k] = starting_value * (1 + rate * x_val[k])  ## simple compounding (not annualised)

    if (show >= 1):
        plt.plot(x_val, y_val_nc, 'b-')

    if (show >= 2):
        plt.plot(x_val, y_val_sc, 'g-')

    if (show >= 3):
        plt.plot(x_val, y_val, 'r-')

    if (show >= 4):
        plt.plot(x_val, y_val_sc_pt, 'm-')

#    if (show >= 5):
#        plt.plot(x_val, y_val_gbm, 'm-')

    max_v = max(y_val) * 1.5
    if (show >= 4):
        max_v = max(y_val_sc_pt) * 1.5

    plt.axis([t_min, t_max, min(y_val)*0.05, max_v])
    plt.title("Compounding Growth for Rate={0}".format(rate))
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()


if __name__ == '__main__':

    rate = 0.075
    vol = 0.2
    show = 1
    compounding_plot(rate, vol, show)
