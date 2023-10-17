import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bpf import run_bpf
from models.sv import SV
from tqdm import tqdm


def get_snp_data(path="data/SP.csv"):
    df = pd.read_csv(path)
    x, r = None, np.array(df["Close"])  # S&P closing index
    y = np.log(r[1:]) - np.log(r[0:-1])  # log returns
    y_trend_removed = y[1:] - y[:-1]  # remove any remaining trend

    return y_trend_removed


y = get_snp_data()
T = len(y)
N = 1000
runs = 10
truth_particles = 50000
np.random.seed(0)
rs_list = ['kl', 'kl-iw', 'multinomial', 'systematic', 'stratified']

s, b, m = 1, 0.01, 0.8
model = SV(sigma=s, beta=b, phi=m)
np.random.seed(0)
truth = run_bpf(y, truth_particles, model=model, resampling_scheme='multinomial', d=1)
x_star = truth['particles']
print("Ground truth marg. log-likelihood", truth["marg_log_likelihood"])
for rs in rs_list:
    np.random.seed(0)
    mse_filtering = []
    mse_predictive = []
    mse_I_1 = []
    marg_log_likelihoods = []
    elbos = []
    ess = []
    tvs = []
    print(rs)

    for r in tqdm(range(runs)):
        out = run_bpf(y, N, model=model, resampling_scheme=rs, adaptive=False, beta=1, d=1)
        q = out['posterior']
        paths = out['B']
        particles = out['particles']
        idx = np.arange(N)
        idx_star = np.arange(truth_particles)
        I_1 = 0
        for t in reversed(range(T)):
            I_1 += ((q @ particles[idx, :, t] - truth['posterior'] @ x_star[idx_star, :, t]) ** 2) / T
            idx = paths[idx, t]
            idx_star = truth['B'][idx_star, t]
        mse_I_1.append(I_1)
        marg_log_likelihoods.append(out['marg_log_likelihood'])
        elbos.append(out["elbo"])
        ess.append(np.mean(out['ESS']))
        tvs.append(np.mean(out['tvs']))
    # plot_paths(particles, paths)

    # print('Resampling scheme: ', rs)
    # # print("Filtering:", np.mean(mse_filtering))
    # # print("Prediction:", np.mean(mse_predictive))
    print("Avg. marg. log-likelihood:", np.mean(marg_log_likelihoods))
    print("Std of marg. log-likelihood estimates:", np.std(marg_log_likelihoods))
    print("Avg. ELBOs: ", np.mean(elbos))
    print("Avg. I_1:", np.mean(mse_I_1), np.std(mse_I_1))
    print("Avg. ESS: ", np.mean(ess))
    print("Avg. TV", np.mean(tvs))
    print()
