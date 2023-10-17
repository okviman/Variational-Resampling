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
    def __init__(self, dt, s=10,r=28,b=8/3):
        # global stats
        self.s = s
        self.r = r
        self.b = b
        self.dt = dt # needs to be small enough for the discretization to be accurate
        self.prior_mean = np.zeros(3)
        self.prior_cov = np.eye(3)

        self.state_noise = 0.5 * self.dt

    def generateData(self, T):

        # Latent state is three dimensional

        xs = np.empty(T)
        ys = np.empty(T)
        zs = np.empty(T)

        # Set initial values
        xs[0], ys[0], zs[0] = self.particle_0(N=1).T.squeeze()

        observations = [xs[0] + npr.normal(0, 1)]

        # Step through "time", calculating the partial derivatives at the current point
        # and using them to estimate the next point
        for i in range(T - 1):
            x_dot, y_dot, z_dot = lorenz_discretized(xs[i], ys[i], zs[i], self.s, self.r, self.b)
            xs[i + 1] = xs[i] + (x_dot * self.dt) + npr.normal(0, self.state_noise)
            ys[i + 1] = ys[i] + (y_dot * self.dt) + npr.normal(0, self.state_noise)
            zs[i + 1] = zs[i] + (z_dot * self.dt) + npr.normal(0, self.state_noise)

            observations.append(xs[i + 1] + npr.normal(0, 1))  # We observe only the first coordinate !

        latent_states = np.vstack((xs, ys, zs))
        observations = np.asarray(observations)

        return latent_states, observations

    def particle_0(self, N):
        # return npr.multivariate_normal(self.prior_mean, self.prior_cov, size=N)
        return npr.normal(0, 1, size=(N, 3))

    def propagate(self, x):
        return x + npr.normal(size=x.shape)

    def log_g(self, x, y):
        return stats.norm.logpdf(x=y, loc=x, scale=1).squeeze()

    def log_f(self, x_current, x_previous):
        if x_previous is not None:
            log_prior = stats.norm.logpdf(x_current, loc=x_previous, scale=1).sum(-1)
        else:
            log_prior = stats.norm.logpdf(x_current, loc=0, scale=1).sum(-1)
        return log_prior

    def log_joint(self, x_current, x_previous, y):
        """
        Computes the log joint probability (input to the Greedy algorithm) at time n,
        log p(x_{n}, y_{n}) = log g(y_n)f(x_n|x_{n-1})
        """
        log_prior = self.log_f(x_current, x_previous)
        ll = self.log_g(x_current, y)
        return log_prior + ll


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    # # Instantiate the model and generate data
    continuous_T = 10
    dt = 0.01
    T = int(continuous_T / dt)
    print(T)


    np.random.seed(0)
    model = Lorenz63(dt=dt)
    X, observations = model.generateData(T)

    original_x_axis = np.arange(0, T, 1)
    new_x_axis = original_x_axis * dt
    #
    plt.plot(new_x_axis, X[0, :])
    plt.show()

    # Plot the first three variables
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    matplotlib.rcParams.update({'font.size': 22})
    ax.set_title('Lorenz 63 (3D Latent Space)')
    ax.plot(X[0, :], X[1, :], X[2, :])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.show()

    #