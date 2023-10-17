import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import logsumexp
from scipy.stats import norm
from resampling import kl, multinomial_resampling, systematic_resampling, stratified_resampling


"""def eval_log_p(x):
    p1 = norm(0, 0.2)
    p2 = norm(0, 0.2)
    pi = np.array([0.3, 0.7])
    return logsumexp([np.log(pi[0]) + p1.logpdf(x), np.log(pi[1]) + p2.logpdf(x)], axis=0)
"""
def eval_log_p(x, pi):
    py_x = norm(0., sigma_y)
    px = pi
    log_joint = py_x.logpdf(x) + px.logpdf(x)
    return log_joint


def plot_resampled_particles(x, title):
    x_linspace = np.linspace(-3, 3, 1000)
    plt.plot(x_linspace, np.exp(eval_log_p(x_linspace, pi) - pi.logpdf(x_linspace)), color='Red', label='$g(y|x)$')
    plt.plot(x_linspace, np.exp(pi.logpdf(x_linspace)), color='Blue', label='$f(x)$')
    if title == 'Samples':
        plt.scatter(x, np.zeros_like(x), color='Black')
    else:
        # sns.kdeplot(x)
        plt.ylabel('')
        plt.scatter(x, np.zeros_like(x), color='Black')
    plt.title(title)
    plt.legend()
    plt.show()


def get_reselbo(idx, log_w_tilde, S):
    unique_idx, multiplicities = np.unique(idx, return_counts=True)
    return (multiplicities / S) @ (log_w_tilde[unique_idx] - np.log(multiplicities))


def get_tv(idx, log_w_tilde, S):
    q = np.zeros(S)
    unique_idx, multiplicities = np.unique(idx, return_counts=True)
    q[unique_idx] = multiplicities / S
    return 0.5 * np.sum(np.abs(q - np.exp(log_w_tilde)))


if __name__ == '__main__':
    np.random.seed(0)
    pi = norm(0, 1)
    N = 1000
    x = pi.rvs(N)
    sigma_y = 0.5
    log_p = eval_log_p(x, pi)

    log_w = log_p - pi.logpdf(x)
    w_tilde = np.exp(log_w - logsumexp(log_w))

    log_p_normed = log_p - logsumexp(log_p)


    matplotlib.rcParams.update({'font.size': 22})
    plot_resampled_particles(x, 'Samples')
    print(logsumexp(log_w) - np.log(N))
    for rs, title in zip([kl, kl, multinomial_resampling, systematic_resampling, stratified_resampling],
                         [r'LB Resampler w. $\pi_{\gamma}$', r'LB Resampler w. $\pi_{w}$', 'Multinomial Resampling', 'Systematic Resampling',
                          'Stratified Resampling']):
        if r'\pi_{w}' in title:
            idx = rs(log_w)
        elif r'gamma' in title:
            idx = rs(log_p)
        else:
            idx = rs(w_tilde)
        x_resampled = x[idx]
        plot_resampled_particles(x_resampled, title)
        print(title)
        """
        print("Using model based target:")
        print("KL = ", get_kl(idx, log_p_normed, S))
        print("TV =", get_tv(idx, log_p_normed, S))
        """
        print("Using importance weighted target:")
        print("ResELBO = ", get_reselbo(idx, log_w, N))
        print("TV =", get_tv(idx, np.log(w_tilde), N))
        # print("KDE estimated KLDs")
        # print(get_kde_kl(x_resampled))
        print()


