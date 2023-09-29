import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.special import logsumexp
from resampling import kl, multinomial_resampling, systematic_resampling, stratified_resampling
from toy_experiment_visualization import get_reselbo, get_tv


def eval_log_p(x):
    p1 = norm(-0.5, 0.2)
    p2 = norm(0.5, 0.2)
    pi = np.array([0.3, 0.7])
    return logsumexp([np.log(pi[0]) + p1.logpdf(x), np.log(pi[1]) + p2.logpdf(x)], axis=0)


np.random.seed(0)
pi = norm(0.5, 1)
n_iterations = 100
N_list = [100, 500, 1000, 5000]
n_N = len(N_list)
resampling_schemes = [kl, kl, multinomial_resampling, systematic_resampling, stratified_resampling]
resampling_names = [r'ELBO Resampler w. $\bar{p}$', r'ELBO Resampler w. $q$', 'Multinomial Resampling', 'Systematic Resampling',
                              'Stratified Resampling']

# last two dimensions are mean and std
reselbo_stats = np.zeros((len(resampling_schemes), n_N, 2))

for s, S in enumerate(N_list):
    reselbos_n = np.zeros((len(resampling_schemes), n_iterations))
    for n in range(n_iterations):
        x = pi.rvs(S)
        log_p = eval_log_p(x)
        log_w = log_p - pi.logpdf(x)
        w = np.exp(log_w - logsumexp(log_w))

        for i, (rs, title) in enumerate(zip(resampling_schemes, resampling_names)):
            if 'q' in title:
                idx = rs(log_w)
            elif r'\bar{p}' in title:
                idx = rs(log_p)
            else:
                idx = rs(w)
            reselbos_n[i, n] = get_reselbo(idx,  log_w, S)
    reselbo_stats[:, s, 0] = reselbos_n.mean(axis=1)
    reselbo_stats[:, s, 1] = reselbos_n.std(axis=1)

matplotlib.rcParams.update({'font.size': 20})
colors = ['Blue', 'Black', 'Red', 'Purple', 'Green']
for i, (rs, title, c) in enumerate(zip(resampling_schemes, resampling_names, colors)):
    x_axis = np.arange(len(N_list))
    plt.plot(x_axis, reselbo_stats[i, :, 0], label=title, color=c)
    plt.fill_between(x_axis, reselbo_stats[i, :, 0] - 0.5 * reselbo_stats[i, :, 1],
                     reselbo_stats[i, :, 0] + 0.5 * reselbo_stats[i, :, 1], alpha=0.3,
                     color=c)
    plt.xticks(x_axis, N_list)
plt.xlabel('$N$')
plt.ylabel('ResELBO')
plt.ylim(-0.45, 0.05)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
