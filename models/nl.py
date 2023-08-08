import numpy as np
import scipy.stats as stats


class NL:
    """
    The model parameters and pdf:s for the Stochastic Volatility model.
    """
    def __init__(self):
        # global stats
        self.x_std = 1.  # std of transition probability
        self.t = 0

    def generateData(self, T):
        x = np.zeros(T)
        y = np.zeros(T)

        x[0] = np.random.normal(size=1)
        y[0] = x[0]**2/20 + np.random.normal(size=1) * np.sqrt(10)

        for t in range(1, T):
            x[t] = x[t - 1]/2 + 25 * x[t - 1]/(1 + x[t - 1]**2) + 8 * np.cos(1.2 * t) + np.random.normal(size=1)
            y[t] = x[t]**2/20 + np.random.normal(size=1) * np.sqrt(10)

        return x, y

    def particle_0(self, N):
        return np.random.normal(size=N)

    def propagate(self, x):
        self.t += 1
        x_next = x/2 + 25 * x/(1 + x**2) + 8 * np.cos(1.2 * self.t) + np.random.normal(size=x.size)
        return x_next

    def log_g(self, x, y):
        return stats.norm.logpdf(y, loc=x**2/20, scale=np.sqrt(10))

    def log_f(self, x_current, x_previous):
        if x_previous is not None:
            mean = x_previous/2 + 25 * x_previous/(1 + x_previous**2) + 8 * np.cos(1.2 * self.t)
            log_prior = stats.norm.logpdf(x_current, loc=mean, scale=1)
        else:
            log_prior = stats.norm.logpdf(x_current, loc=0, scale=1)
        return log_prior

    def log_joint(self, x_current, x_previous, y):
        """
        Computes the log joint probability (input to the Greedy algorithm) at time n,
        log p(x_{n}, y_{n}) = log g(y_n)f(x_n|x_{n-1})
        """
        if x_previous is not None:
            mean = x_previous/2 + 25 * x_previous/(1 + x_previous**2) + 8 * np.cos(1.2 * self.t)
            log_prior = stats.norm.logpdf(x_current, mean, 1)
        else:
            log_prior = stats.norm.logpdf(x_current, loc=0, scale=1)
        ll = self.log_g(x_current, y)
        return log_prior + ll