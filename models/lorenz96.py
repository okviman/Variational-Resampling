import numpy.random as npr
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# def lorenz96_discretized(X, F):
#     '''
#     Discretized Lorenz 96 dynamics with a simple Euler scheme
#
#     Given:
#        X: state vector of the Lorenz 96 model
#        F: forcing term
#     Returns:
#        X_dot: derivatives at the current state
#     '''
#     D = len(X)
#     X_dot = np.empty(D)
#
#     for i in range(D):
#         X_dot[i] = (X[(i + 1) % D] - X[i - 2]) * X[i - 1] - X[i] + F
#
#     return X_dot

def lorenz96_discretized(X, F):
    D = len(X)
    ip1 = np.arange(1, D + 1) % D
    im1 = np.arange(-1, D - 1) % D
    im2 = np.arange(-2, D - 2) % D
    X_dot = (X[ip1] - X[im2]) * X[im1] - X + F
    return X_dot



class Lorenz96:
    """
    The model parameters and pdf:s for the Lorenz 96 model.
    """

    def __init__(self, D, dt,  F=8):
        self.F = F
        self.D = D
        self.dt = dt
        self.prior_mean = np.zeros(D)
        self.prior_cov = np.eye(D)
        self.number_of_observed_coordinates = int(D / 2)

        self.state_noise = 0.5 * self.dt

    def generateData(self, T):
        X = np.empty((self.D, T))
        X[:, 0] = self.particle_0(N=1).T.squeeze()
        observations = [X[:self.number_of_observed_coordinates, 0] + npr.normal(0, 1, size=self.number_of_observed_coordinates)]

        for i in range(T - 1):
            X_dot = lorenz96_discretized(X[:, i], self.F)
            X[:, i + 1] = X[:, i] + (X_dot * self.dt) + npr.normal(0, self.state_noise, size=self.D)
            # Without noise only for plotting !
            # X[:, i + 1] = X[:, i] + (X_dot * self.dt)

            observations.append(X[:self.number_of_observed_coordinates, i + 1] + npr.normal(0, 1, size=self.number_of_observed_coordinates))

        observations = np.asarray(observations)
        return X, observations

    def particle_0(self, N):
        return npr.normal(0, 1, size=(N, self.D))

    def propagate(self, x):
        return x + npr.normal(0, self.state_noise, size=x.shape)

    def log_g(self, x, y):
        return multivariate_normal.logpdf( x=y, mean=x[:self.number_of_observed_coordinates], cov=np.eye(self.number_of_observed_coordinates) )

    def log_f(self, x_current, x_previous):
        if x_previous is not None:
            log_prior = stats.norm.logpdf(x_current, loc=x_previous, scale=self.state_noise).sum(-1)
        else:
            log_prior = stats.norm.logpdf(x_current, loc=0, scale=self.state_noise).sum(-1)
        return log_prior

    def log_joint(self, x_current, x_previous, y):
        log_prior = self.log_f(x_current, x_previous)
        ll = self.log_g(x_current, y)
        return log_prior + ll


# # Instantiate the model and generate data
continuous_T = 6
dt = 0.01
D = 100
T = int(continuous_T / dt)
print(T)


model = Lorenz96(D=D,dt=dt)
X, observations = model.generateData(T)

# original_x_axis = np.arange(0, T, 1)
# new_x_axis = original_x_axis * dt
#
# plt.plot(new_x_axis, X[0, :])
# plt.show()

# Plot the first three variables
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(X[0, :], X[1, :], X[2, :])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.show()

#
