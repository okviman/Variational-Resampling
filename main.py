from models.sv import SV
from models.nl import NL
from models.lorenz63 import Lorenz63
from models.lorenz96 import Lorenz96

import numpy as np
from bpf import run_bpf
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


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


continuous_T = 10
dt = 0.01
T = int(continuous_T / dt)

runs = 10
model, d = SV(), 1
# model, d = NL(), 1
# model, d = Lorenz63(dt=dt), 3
truth_particles = 50000
common_settings = "continuousT_{}_dt_{}_nruns_{}_truthparticles_{}".format(continuous_T, dt, runs, truth_particles)

for D in tqdm([d]):
    # N = int(D * 10)
    N = 1000
    # model = Lorenz96(D=D, dt=dt)
    np.random.seed(0)
    data = [model.generateData(T) for _ in range(1)]

    rs_list = ['kl', 'kl-iw', 'multinomial', 'systematic', 'stratified']
    (x, y) = data[0]
    truth = run_bpf(y, truth_particles, model=model, resampling_scheme='multinomial', adaptive=False, d=D)
    x_star = truth['particles']
    print("Ground truth marg. log-likelihood", truth["marg_log_likelihood"])
    for rs in rs_list:
        np.random.seed(0)
        mse_filtering = []
        mse_predictive = []
        mse_I_0 = []  # I_0 = MSE^* in the paper
        mse_I_1 = []  # I_1 = MSE
        marg_log_likelihoods = []
        elbos = []
        ess = []
        tvs = []

        for r in tqdm(range(runs)):
            (x, y) = data[0]
            out = run_bpf(y, N, model=model, resampling_scheme=rs, adaptive=False, d=D)

            q = out['posterior']
            paths = out['B']
            particles = out['particles']
            idx = np.arange(N)
            idx_star = np.arange(truth_particles)
            I_0 = 0
            I_1 = 0
            for t in reversed(range(T)):
                I_0 += ((q @ particles[idx, :, t] - x[:, t]) ** 2) / T
                I_1 += ((q @ particles[idx, :, t] - truth['posterior'] @ x_star[idx_star, :, t]) ** 2) / T
                idx = paths[idx, t]
                idx_star = truth['B'][idx_star, t]
            mse_I_0.append(I_0)
            mse_I_1.append(I_1)
            marg_log_likelihoods.append(out['marg_log_likelihood'])
            elbos.append(out["elbo"])
            ess.append(np.mean(out['ESS']))
            tvs.append(np.mean(out['tvs']))
        # plot_paths(particles, paths)

        # print('Resampling scheme: ', rs)
        print("Avg. marg. log-likelihood:", np.mean(marg_log_likelihoods))
        print("Std of marg. log-likelihood estimates:", np.std(marg_log_likelihoods))
        print("Avg. ELBOs: ", np.mean(elbos))
        print("Avg. I_0:", np.mean(mse_I_0), np.std(mse_I_0))
        print("Avg. I_1:", np.mean(mse_I_1), np.std(mse_I_1))
        print("Avg. ESS: ", np.mean(ess))
        print("Avg. TV", np.mean(tvs))
        print()

        np.save('./results/{}/avgloglik_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings),
                np.mean(marg_log_likelihoods))
        np.save('./results/{}/mseloglikwrtgroundtruth_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings),
                np.mean((marg_log_likelihoods - truth["marg_log_likelihood"]) ** 2))
        np.save('./results/{}/biasaquaredloglik_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings),
                (np.mean(marg_log_likelihoods) - truth["marg_log_likelihood"]) ** 2)
        np.save('./results/{}/varianceloglik_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings),
                np.var(marg_log_likelihoods))
        np.save('./results/{}/avgI-0_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings),
                np.mean(mse_I_0))
        np.save('./results/{}/stdI-0_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings), np.std(mse_I_0))
        np.save('./results/{}/avgI-1_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings),
                np.mean(mse_I_1))
        np.save('./results/{}/stdI-1_dimensions_{}_N_{}_{}.npy'.format(rs, str(D), N, common_settings), np.std(mse_I_1))

        del out
        gc.collect()

