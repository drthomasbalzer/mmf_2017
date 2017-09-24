import numpy as np
import math

######
## PDF of normal distribution
######

def normal_pdf(x, mu, sigma_sq):
    y = 1. / np.sqrt(2 * np.pi * sigma_sq) * np.exp(- 0.5 * np.power(x - mu, 2) / sigma_sq)
    return y

def standard_normal_pdf(x):
    return normal_pdf(x, 0, 1)

######
## Exponential distribution with parameter $\lambda > 0$
######

######
## pdf and inverse distribution function of $Exp(\lambda)$-distribution
######

def exponential_pdf(_lambda, x):

    if (x < 0):
        return 0.
    else:
        return _lambda * np.exp(-_lambda * x)

def exponential_inverse_cdf(_lambda, x):

    if (x < 0):
        return 0.
    else:
        return -1./_lambda * np.log(1-x)

#####
## Details of a random variable $X$ with $\mathbb{P}(X = 1) = p = 1 - \mathbb{P}(X = 0)$
#####

def binomial_inverse_cdf(p, x):

    if (x < 1 - p):
        return 0.
    else:
        return 1.

def binomial_expectation(p):
    return p

def binomial_variance(p):
    return p * (1 - p)

#####
## Details of a random variable $X$ with $\mathbb{P}(X = 1) = p = 1 - \mathbb{P}(X = -1)$
#####

def symmetric_binomial_inverse_cdf(p, x):

    if (x < 1-p):
        return -1.
    else:
        return 1.

def symmetric_binomial_expectation(p):
    return 2 * p - 1

def symmetric_binomial_variance(p):
    return 4 * p * (1 - p)

###########
##
## Discrete Probability Distributions
##
###########

#########
### returns the probability that the Poisson distribution takes value $k \in \mathbb{N}_0$.
#########

def poisson_pdf(_lambda, k):


    p = np.exp(-_lambda) * np.power(_lambda, k) / math.factorial(k)
    return p

#########
### returns the probability that the Binomial distribution $B(n,p)$ distribution takes value $k \in \{0,\ldots, n\}$.
#########

def binomial_pdf(p, k, n):

    ### probability calculated "by hand" - is available in numpy 'off-the-shelf'

    p = np.power(1-p, n-k) * np.power(p, k) * math.factorial(n) / math.factorial(k) / math.factorial(n-k)
    return p

