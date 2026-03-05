import numpy as np

def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def bernoulli_dist(x, w):
    return (w**x) * (1-w)**(1-x)

def log_normal_dist(x, mean, sd):
    log_prob_density = -0.5 * np.log(2 * np.pi * sd**2) - 0.5 * ((x - mean)**2) / (sd**2)
    return log_prob_density


#Old notebook code

# import sympy as sp
# from sympy import Array
# from sympy.plotting import plot3d,  plot
# from partialDerivator import cost_function, gradient_descent, cost_function

# x, sigma, mu = sp.symbols('x sigma mu')
# f = cost_function(x_bold=samples_list, mu=mu, sigma=1)
# plot3d(f, (x, -20, 20), (mu, -20, 20))
# x_bold = samples_list
# best_mu = gradient_descent(func=f, var=mu, datapoints=x_bold, learning_rate=0.01, iterations=100)

# print(f"Best mu: {best_mu}")
# initial_loss = cost_function(x_bold=samples_list, mu=0, sigma=1)
# final_loss = cost_function(x_bold=samples_list, mu=best_mu, sigma=1)
# print(f"Initial loss: {float(initial_loss)}, Final loss: {float(final_loss)}")





# --- 

# import numpy as np
# from sympy import symbols, exp, sqrt, pi, log, lambdify
# from partialDerivator import cost_function, gradient_descent, compute_loglikelihood, get_normal_density_value

# x, mu, sigma = symbols('x mu sigma', real=True)
# def normal_density(x, mu, sigma):
#     coeff = 1 / (sqrt(2 * pi) * sigma)
#     exponent = exp(-(x - mu)**2 / (2 * sigma**2))
#     return coeff * exponent
# normal_dist = (1 / (sqrt(2*pi)*sigma)) * exp(-(x - mu)**2 / (2*sigma**2))

# logpdf = log(normal_dist)

# logpdf_func = lambdify(
#     (x, mu, sigma),
#     logpdf,
#     'numpy'
# )

# mu_true = 2
# sigma = 1
# n = 500

# x_samples = np.random.normal(mu_true, sigma, n)
# x_samples = samples_list

# mu_1 = 2
# mu_2 = 7

# logL_mu2 = np.sum(logpdf_func(x_samples, mu_1, sigma))
# logL_mu7 = np.sum(logpdf_func(x_samples, mu_2, sigma))

# normal_dist1 = normal_dist.subs({sigma:1, mu:mu_1})
# normal_dist2 = normal_dist.subs({sigma:1, mu:mu_2})

# print( compute_loglikelihood(x_samples, normal_density(x_samples,mu,sigma), ({'mu':mu_1, 'sigma':1})))
# print( compute_loglikelihood(x_samples, normal_density(x_samples,mu,sigma), ({'mu':mu_2, 'sigma':1})))

# print("log L(mu=2):", logL_mu2)
# print("log L(mu=7):", logL_mu7)
# print("Diferença:", logL_mu2 - logL_mu7)

# print(get_normal_density_value(x_samples[1], sigma, mu_1))
# print(get_normal_density_value(x_samples[1], sigma, mu_2))