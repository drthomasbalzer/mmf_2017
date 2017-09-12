import numpy as np
import math
import matplotlib.pyplot as plt

def normal_pdf(x, mu, sigma_sq):
    y = 1. / np.sqrt(2 * np.pi * sigma_sq) * np.exp(- 0.5 * np.power(x - mu, 2) / sigma_sq)
    return y

def normal_distribution_plot(mu, sigma_sq):

    t_min = -5.
    t_max = 5.
    n = 1000
    step = (t_max - t_min) / n

    x_val = range(n)  # values on the x axis
    y_val = range(n)  # y_values - normal distribution
    y_val_sn = range(n)  # y_values - standard normal


    for k in range(n):
        x_val[k] = t_min + step * k
        y_val[k] = normal_pdf(x_val[k], mu, sigma_sq)
        y_val_sn[k] = normal_pdf(x_val[k], 0, 1)

    #######
    ### prepare and show plot
    ###
    plt.plot(x_val, y_val, 'r-')
    plt.plot(x_val, y_val_sn, 'g--')
    plt.axis([t_min, t_max, min(y_val)*.9, max(y_val)*1.5])
    plt.title("Normal PDF")
    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.show()
    ###
    #######


if __name__ == '__main__':

    mu = -1
    sigma_sq = 1
    normal_distribution_plot(mu, sigma_sq)