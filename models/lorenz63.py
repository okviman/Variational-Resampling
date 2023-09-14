import numpy.random as npr
import numpy as np
import scipy.stats as stats

def lorenz_discretized(x, y, z, s, r, b):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


class Lorenz63:
    """
    The model parameters and pdf:s for the Lorenz 63 model. Default params from Wikipedia page
    """
    def __init__(self, s=10,r=28,b=2.667):
        # global stats
        self.s = s
        self.r = r
        self.b = b
        self.dt = 1e-3 # needs to be small enough for the discretization to be accurate
        self.prior_mean = np.zeros(3)
        self.prior_cov = np.eye(3)

    def generateData(self, T):

        observations = []

        # Latent state is three dimensional

        xs = np.empty(T + 1)
        ys = np.empty(T + 1)
        zs = np.empty(T + 1)

        # Set initial values
        xs[0], ys[0], zs[0] = tuple(self.particle_0(N=1))

        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(T):
            x_dot, y_dot, z_dot = lorenz_discretized(xs[i], ys[i], zs[i], self.s, self.r, self.b)
            xs[i + 1] = xs[i] + (x_dot * self.dt) + npr.randn()
            ys[i + 1] = ys[i] + (y_dot * self.dt) + npr.randn()
            zs[i + 1] = zs[i] + (z_dot * self.dt) + npr.randn()

            observations.append(xs[i + 1] + npr.normal(0, 1)) # We observe only the first coordinate !

        latent_states = np.vstack((xs, ys, zs)).T
        observations = np.asarray(observations)

        return latent_states, observations

    def particle_0(self, N):
        return npr.multivariate_normal(self.prior_mean, self.prior_cov, size=N)

    def propagate(self, x):
        return x + npr.normal(size=x.size)

    def log_g(self, x, y):
        return stats.norm.logpdf(x=y, loc=x, scale=1)

    def log_f(self, x_current, x_previous):
        if x_previous is not None:
            log_prior = stats.norm.logpdf(x_current, loc=x_previous, scale=1)
        else:
            log_prior = stats.norm.logpdf(x_current, loc=0, scale=1)
        return log_prior

    def log_joint(self, x_current, x_previous, y):
        """
        Computes the log joint probability (input to the Greedy algorithm) at time n,
        log p(x_{n}, y_{n}) = log g(y_n)f(x_n|x_{n-1})
        """
        log_prior = self.log_f(x_current, x_previous)
        ll = self.log_g(x_current, y)
        return log_prior + ll