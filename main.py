from scipy.stats import norm
from scipy.special import logsumexp
import numpy as np
from resampling import kl, systematic_resampling, tv
import matplotlib.pyplot as plt


def idx_to_pmf(idx, multiplicities, S):
    pmf = np.zeros(S)
    pmf[idx] = multiplicities / S
    return pmf


def get_tv(idx, log_w_tilde, S):
    unique_idx, multiplicities = np.unique(idx, return_counts=True)
    return 0.5 * np.abs(np.exp(log_w_tilde[unique_idx]) - multiplicities / S).sum()


def get_elbo(idx, log_joint, S):
    unique_idx, multiplicities = np.unique(idx, return_counts=True)
    return (multiplicities / S) @ (log_joint[unique_idx] - np.log(multiplicities / S))


tv_elbos = []
tv_tvs = []

kl_elbos = []
kl_tvs = []

systematic_elbos = []
systematic_tvs = []

np.random.seed(0)
p = norm(0, 1.5)
log_Z = norm(0, 0.001).logpdf(0)

S_list = [50, 100, 500, 1000, 10000, 50000]
for S in S_list:
    x = np.random.normal(0, 10, S)
    log_joint = p.logpdf(x)
    log_w_tilde = log_joint - logsumexp(log_joint)

    idx = kl(log_joint)
    # q_ddot_kl = idx_to_pmf(unique_idx, multiplicities, S)
    kl_elbos.append(get_elbo(idx, log_w_tilde, S))
    kl_tvs.append(get_tv(idx, log_w_tilde, S))

    idx = tv(np.exp(log_w_tilde))
    tv_elbos.append(get_elbo(idx, log_w_tilde, S))
    tv_tvs.append(get_tv(idx, log_w_tilde, S))

    idx = systematic_resampling(np.exp(log_w_tilde))
    systematic_elbos.append(get_elbo(idx, log_w_tilde, S))
    systematic_tvs.append(get_tv(idx, log_w_tilde, S))

plt.plot(tv_tvs, 'r')
plt.plot(kl_tvs, 'b')
plt.plot(systematic_tvs, 'm')
plt.xticks(np.arange(len(S_list)), S_list)
plt.ylabel('TV')
plt.xlabel('$S$')
plt.show()

plt.plot(tv_elbos, 'r')
plt.plot(kl_elbos, 'b')
plt.plot(systematic_elbos, 'm')
plt.xticks(np.arange(len(S_list)), S_list)
plt.ylabel('ELBO')
plt.xlabel('$S$')
plt.show()

"""
ordered_idx = np.argsort(x)
plt.plot(x[ordered_idx], np.exp(log_joint[ordered_idx]))
plt.show()
ordered_idx = np.argsort(x[idx])
plt.plot(x[idx][ordered_idx], np.exp(log_joint[idx][ordered_idx]), 'r')
plt.plot(x[idx][ordered_idx], q_ddot_kl[ordered_idx], 'b')
plt.show()
"""
