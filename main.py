from models.sv import SV
from models.nl import NL
import numpy as np
from bpf import run_bpf
import matplotlib.pyplot as plt


def plot_paths(particles, B):
    X = np.zeros_like(particles)
    indx = B[:, -1]
    for t in reversed(range(1, T)):
        X[:, t] = particles[indx, t]
        indx = B[indx, t - 1]
    X[:, 0] = particles[indx, 0]
    for n in range(N):
        plt.plot(X[n], color='Blue')
    plt.plot(x, color='Red')
    plt.show()


T = 500
N = 1000
runs = 1
model = SV()
# model = NL()

np.random.seed(2)
data = [model.generateData(T) for _ in range(runs)]

# n_test = 100
# test_x_set = np.zeros((n_test, T))
# for i in range(n_test):
#     test_x_set[i] = model.generateData(T)[0]

# rs_list = ['kl', 'multinomial', 'systematic', 'stratified', 'tv', 'cubo']
rs_list = ['kl', 'multinomial']
(x, y) = data[0]
truth_particles = 50000
truth = run_bpf(y, truth_particles, model=model, resampling_scheme='multinomial', adaptive=False, beta=1)
x_star = truth['particles']
for rs in rs_list:
    np.random.seed(0)
    mse_filtering = []
    mse_predictive = []
    mse_smoothing = []
    marg_log_likelihoods = []
    elbos = []

    for r in range(runs):
        (x, y) = data[r]
        out = run_bpf(y, N, model=model, resampling_scheme=rs, adaptive=False, beta=1)
        x_star_filtering = out['filtering'][0, :]
        x_star_predictive = out['predictions'][0, :]
        mse_filtering.append(np.mean((x_star_filtering[:-1] - x[:-1]) ** 2))
        mse_predictive.append((np.mean((x_star_predictive - x) ** 2)))

        q = out['posterior']
        paths = out['B']
        particles = out['particles']
        idx = np.arange(N)
        idx_star = np.arange(truth_particles)
        mse_s = 0
        for t in reversed(range(T)):
            mse_s += ((q @ particles[idx, t] - truth['posterior'] @ x_star[idx_star, t]) ** 2) / T
            # mse_s += ((q @ particles[idx, t] - x[t]) ** 2) / T
            idx = paths[idx, t]
            idx_star = truth['B'][idx_star, t]
        mse_smoothing.append(mse_s)
        marg_log_likelihoods.append(out['marg_log_likelihood'])
        elbos.append(out["elbo"])
    plot_paths(particles, paths)

    print('Resampling scheme: ', rs)
    print("Filtering:", np.mean(mse_filtering))
    print("Prediction:", np.mean(mse_predictive))
    print("Avg. marg. log-likelihood:", np.mean(marg_log_likelihoods))
    print("Avg. ELBOs: ", np.mean(elbos))
    print("Avg. Smoothing:", np.mean(mse_smoothing))
    print()

    # plt.plot(x_star, label=rs)
# plt.legend()
# plt.show()
