import numpy.random as npr
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def lorenz96_discretized(X, F):
    '''
    Discretized Lorenz 96 dynamics with a simple Euler scheme

    Given:
       X: state vector of the Lorenz 96 model
       F: forcing term
    Returns:
       X_dot: derivatives at the current state
    '''
    D = len(X)
    X_dot = np.empty(D)

    for i in range(D):
        X_dot[i] = (X[(i + 1) % D] - X[i - 2]) * X[i - 1] - X[i] + F

    return X_dot


class Lorenz96:
    """
    The model parameters and pdf:s for the Lorenz 96 model.
    """

    def __init__(self, F=8, D=10, dt=0.01):
        self.F = F
        self.D = D
        self.dt = dt
        self.prior_mean = np.zeros(D)
        self.prior_cov = np.eye(D)

    def generateData(self, T):
        X = np.empty((self.D, T))
        X[:, 0] = self.particle_0(N=1).T.squeeze()
        observations = [X[0, 0] + npr.normal(0, 1)]

        for i in range(T - 1):
            X_dot = lorenz96_discretized(X[:, i], self.F)
            # X[:, i + 1] = X[:, i] + (X_dot * self.dt) + npr.normal(0, 1, size=self.D)
            X[:, i + 1] = X[:, i] + (X_dot * self.dt)

            observations.append(X[0, i + 1] + npr.normal(0, 1))

        observations = np.asarray(observations)
        return X, observations

    def particle_0(self, N):
        return npr.normal(0, 1, size=(N, self.D))

    def propagate(self, x):
        return x + npr.normal(size=x.shape)

    def log_g(self, x, y):
        return stats.norm.logpdf(x=y, loc=x[0], scale=1)

    def log_f(self, x_current, x_previous):
        if x_previous is not None:
            log_prior = stats.norm.logpdf(x_current, loc=x_previous, scale=1).sum(-1)
        else:
            log_prior = stats.norm.logpdf(x_current, loc=0, scale=1).sum(-1)
        return log_prior

    def log_joint(self, x_current, x_previous, y):
        log_prior = self.log_f(x_current, x_previous)
        ll = self.log_g(x_current, y)
        return log_prior + ll


# Instantiate the model and generate data
model = Lorenz96()
X, observations = model.generateData(10000)

print(X.shape)
# Plot the first three variables
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(X[0, :], X[1, :], X[2, :])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.show()
