import sympy as sp
import numpy as np
from sympy import sqrt,ln, pi, lambdify
from sympy.plotting import plot3d

def loss_function(x, sigma, mu):
    reconstruction_loss = -ln(sqrt(2 * pi)) - .5 * (x - mu)**2
    kl_divergence = 0.5 * (ln(sigma**2) - mu**2 - sigma**2 + 1)
    return reconstruction_loss + kl_divergence

def get_normal_density_value(x, sigma, mu):
    coeff = 1 / (sqrt(2 * pi) * sigma)
    exponent = sp.exp(-0.5 * ((x - mu) / sigma)**2)
    return float(coeff * exponent)

def cost_function(x_bold, sigma, mu):
    total_cost = 0
    for x in x_bold:
        total_cost += -loss_function(x=x, sigma=sigma, mu=mu)
    return total_cost

def gradient_descent_step(func, var, learning_rate, var_value, data_point):
    gradient = sp.diff(func, var)
    gradient = gradient.subs({var:var_value, 'x': data_point})
    updated_var_value = var_value - learning_rate * gradient
    return updated_var_value

def gradient_descent(func, datapoints, var, learning_rate, iterations):
    var_value = 0
    for _ in range(iterations):
        for data in datapoints:
            var_value = gradient_descent_step(func, var, learning_rate, var_value, data)
    return var_value

def compute_partial_derivative(func, var):
    return sp.diff(func, var)

def compute_loglikelihood(x_bold, dist, parameters):
    parameters['x'] = x_bold
    x_bold = np.array(x_bold)
    log_dist = sp.log(dist)
    log_dist = lambdify(
        list(parameters.keys()),
        log_dist,
        'numpy'
    )

    return np.sum(log_dist(**parameters))

def compute_posterior(x_bold, prior, likelihood, parameters, evidence):
    parameters['x'] = x_bold
    x_bold = np.array(x_bold)
    
    log_prior = sp.log(prior)
    log_likelihood = sp.log(likelihood)
    log_evidence = sp.log(evidence)

    log_prior = lambdify(
        list(parameters.keys()),
        log_prior,
        'numpy'
    )

    log_likelihood = lambdify(
        list(parameters.keys()),
        log_likelihood,
        'numpy'
    )

    log_evidence = lambdify(
        list(parameters.keys()),
        log_evidence,
        'numpy'
    )
    
    log_prior = np.sum(log_prior(**parameters))
    log_likelihood = np.sum(log_likelihood(**parameters))
    log_evidence = np.sum(log_evidence(**parameters))

    return (log_prior + log_likelihood)/log_evidence

def plot_function(func, var, var_range):
    free_variables = []
    if func.free_symbols.__len__() > 1:
        for free_var in func.free_symbols:
            free_variables.append(free_var)
        plot3d( func, 
               (free_variables[0], -10, 10),
               (free_variables[1], -10, 10),
               xlabel=free_variables[0], 
               ylabel=free_variables[1], 
               zlabel=f'f({free_variables[0]}, {free_variables[1]})' 
               )
    else:
        sp.plot(func, (var, var_range[0], var_range[1]), title=f'Plot of {func}', ylabel='f(x)', xlabel=str(var))
