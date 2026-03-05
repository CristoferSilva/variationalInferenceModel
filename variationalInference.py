import numpy as np

def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def bernoulli_dist(x, w):
    return w**x * (1-w)**(1-x)