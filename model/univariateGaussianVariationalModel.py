import warnings
import numpy as np
from sympy import sqrt, ln, pi, exp, sp


class UnivariateGaussianVariationalModel:
    def __init__(self, sigma=1, mu=1):
        self.sigma = sigma
        self.mu = mu
        self.is_fitted = False

    def loss_function(self, x, sigma, mu):
        reconstruction_loss = -ln(sqrt(2 * pi)) - 0.5 * (x - mu) ** 2
        kl_divergence = +0.5 * (ln(sigma**2) - mu**2 - sigma**2 + 1)
        return reconstruction_loss + kl_divergence

    def cost_function(self, x_bold, mu, sigma=1):
        total_cost = 0
        for x in x_bold:
            total_cost += self.loss_function(x=x, sigma=sigma, mu=mu)
        return total_cost

    def __gradient_descent_step(
        self, func, partial_var, learning_rate, var_value, data_point
    ):
        gradient = sp.diff(func, partial_var)
        gradient = gradient.subs({partial_var: var_value, "x": data_point})
        updated_parameter_value = var_value - learning_rate * gradient
        return updated_parameter_value

    def fit(
        self, x_bold_train, x_bold_val=None, learning_rate=0.01, epochs=100, patience=5
    ):
        parameters = {
            "mu": sp.symbols("mu", real=True),
            # "sigma": sp.symbols("sigma", real=True),
        }

        for param in parameters:
            current_parameter_value = self.__dict__[param]
            for _ in range(epochs):
                for x in x_bold_train:
                    current_parameter_value = self.__gradient_descent_step(
                        func=self.cost_function(x_bold=x_bold_train, **parameters),
                        partial_var=parameters[param],
                        learning_rate=learning_rate,
                        var_value=current_parameter_value,
                        data_point=x,
                    )

                if x_bold_val is not None and self.__has_improved(
                    x_bold_val,
                    partial_var=parameters[param],
                    new_var_value=current_parameter_value,
                    old_var_value=self.__dict__[param],
                ):
                    self.__dict__[param] = current_parameter_value
                else:
                    patience -= 1
                if patience == 0:
                    break

        self.is_fitted = True

    def __has_improved(self, x_bold, partial_var, new_var_value, old_var_value):

        new_value_likelihood = sum(
            float(
                self.__get_normal_density_value(x, sigma=1, mu=partial_var).subs(
                    partial_var, new_var_value
                )
            )
            for x in x_bold
        )

        old_value_likelihood = sum(
            float(
                self.__get_normal_density_value(x, sigma=1, mu=partial_var).subs(
                    partial_var, old_var_value
                )
            )
            for x in x_bold
        )

        return new_value_likelihood > old_value_likelihood

    def predict(self, x_bold):
        if not self.is_fitted:
            raise Warning("Model is not fitted yet.")

        densities = []
        for x in x_bold:
            density = self.__get_normal_density_value(x, self.sigma, self.mu)
            densities.append(density)
        return densities

    def __get_normal_density_value(self, x, sigma, mu):
        return (1 / (sqrt(2 * pi * sigma**2))) * exp(
            -0.5 * ((x - mu) ** 2) / (sigma**2)
        )

    def log_normal_dist(self, x, mu, sigma):
        log_prob_density = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (
            (x - mu) ** 2
        ) / (sigma**2)
        return log_prob_density

    def get_parameters(self):
        if not self.is_fitted:
            warnings.warn("Model is not fitted yet.")
        return self.mu, self.sigma
