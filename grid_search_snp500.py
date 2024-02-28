import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bpf import run_bpf
from models.sv import SV
from tqdm import tqdm


def get_snp_data(T=2011, path="data/SP.csv"):
    df = pd.read_csv(path)
    x, r = None, np.array(df["Close"])  # S&P closing index
    y = np.log(r[1:]) - np.log(r[0:-1])  # log returns
    y_trend_removed = y[1:] - y[:-1]  # remove any remaining trend
    y_select = y_trend_removed[0:T]
    return y_select


T = 2011
y = get_snp_data(T)
N = 10000

sigmas = [0.01, 0.1, 0.5, 1, 1.5, 2]
betas = [0.01, 0.1, 0.5, 1, 1.5, 2]
ms = [0.01, 0.1, 0.2, 0.5, 0.6, 0.8, 0.9]

params = []
mlls = []
for i in tqdm(range(len(sigmas))):
    s = sigmas[i]
    for ii in range(len(sigmas)):
        b = betas[ii]
        for iii in range(len(sigmas)):
            m = ms[iii]
            params.append((s, b, m))
            model = SV(sigma=s, beta=b, phi=m)
            truth = run_bpf(y, N, model=model, resampling_scheme='multinomial', d=1)
            mlls.append(truth['marg_log_likelihood'])

idx = np.argmax(mlls)
print(mlls[idx])
print(params[idx])
np.save('./results/grid_search/mlls', np.array(mlls))
np.save('./results/grid_search/params', np.array(params))


