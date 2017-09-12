import matplotlib.pyplot as plt

def bayes_probability(p_se, p_sp):

    #######
    ## calculates Bayes probability
    ##      Example: p_se being the probability of being diagnosed with a sickness, when having the sickness
    ##      Example: p_sp being the probability of being not diagnosed with a sickness, when not having the sickness
    ##      Resulting plot contains the likelihood of being sick conditional on a positive test outcome
    #######

    t_min = 0.0
    t_max = .10
    n = 100
    step = (t_max - t_min) / n

    x_val = range(n)  # values on the x axis
    y_val = range(n)  # y_values - Bayes Probability


    for k in range(n):
        x_val[k] = t_min + step * k
        y_val[k] = x_val[k] * p_se / (x_val[k] * p_se + (1-x_val[k])*(1-p_sp))

    #######
    ### prepare and show plot
    ###
    plt.plot(x_val, y_val, 'r-')
    plt.axis([t_min, t_max, min(y_val)*.9, max(y_val)*1.1])
    plt.title("Likelihood of Sickness under Positive Test Result")
    plt.xlabel("Occurrence")
    plt.ylabel("Likelihood")
    plt.show()
    ###
    #######

if __name__ == '__main__':

    p_se = 0.99
    p_sp = 0.99
    bayes_probability(p_se, p_sp)
