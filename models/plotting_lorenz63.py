import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

s = 10
r = 28
b = 2.667


def lorenz(x, y, z,):
	'''
	Given:
	   x, y, z: a point of interest in three dimensional space
	   s, r, b: parameters defining the lorenz attractor
	Returns:
	   x_dot, y_dot, z_dot: values of the lorenz attractor's partial
		   derivatives at the point x, y, z
	'''
	x_dot = s*(y - x)
	y_dot = r*x - y - x*z
	z_dot = x*y - b*z
	return x_dot, y_dot, z_dot


dt = 0.001
num_steps = 1000



# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)



# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
	x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
	xs[i + 1] = xs[i] + (x_dot * dt)
	ys[i + 1] = ys[i] + (y_dot * dt)
	zs[i + 1] = zs[i] + (z_dot * dt)


# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=3.)
ax.set_xlabel(r"$x_1$ ",labelpad=40)
ax.set_ylabel(r"$x_2$ ",labelpad=40)
ax.set_zlabel(r"$x_3$ ",labelpad=40)
# ax.set_title("Lorenz 63 noiseless trajectory")

plt.show()