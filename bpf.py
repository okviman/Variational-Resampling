from resampling import multinomial_resampling, stratified_resampling, kl, systematic_resampling
import numpy as np
from scipy.special import logsumexp


def run_bpf(y, N, model, resampling_scheme='multinomial', adaptive=False, d=1):
    # Bootstrap Particle Filter (BPF)
    T = len(y)
    d_y = 1

    particles = np.zeros((N, d, T))
    normalized_weights = np.zeros((N, T))
    B = np.zeros((N, T))
    ESS = np.zeros(T)
    log_joint = np.zeros((N, T))
    log_weights = np.zeros((N, T))
    log_l_data = np.zeros((N, T))
    log_l_latent = np.zeros((N, T))
    marg_log_likelihood = 0
    tvs = np.zeros(T)

    if resampling_scheme.lower() in 'multinomial':
        resampling = multinomial_resampling
    elif resampling_scheme.lower() in 'stratified':
        resampling = stratified_resampling
    elif resampling_scheme.lower() in 'systematic':
        resampling = systematic_resampling
    else:
        resampling = kl

    particles[..., 0] = model.particle_0(N)
    log_g_t = model.log_g(x=particles[:, :d_y, 0], y=y[0])
    log_l_data[:, 0] = log_g_t
    log_f_t = model.log_f(particles[..., 0], None)
    log_l_latent[:, 0] = log_f_t
    log_joint[:, 0] = log_l_data[:, 0] + log_l_latent[:, 0]
    normalized_weights[:, 0], log_weights[:, 0] = update_weights(log_weights=log_weights[:, 0], log_g_t=log_g_t)
    new_ancestors = list(range(N))
    B[:, 0] = new_ancestors

    # == Stats == #
    marg_log_likelihood += logsumexp(log_weights[:, 0] - np.log(N))
    elbo = 0

    for t in range(1, T):
        # == Resampling == #
        ESS[t - 1] = 1 / np.sum(normalized_weights[:, t - 1] ** 2)
        if resample_criterion(adaptive, ESS[t - 1], N):
            if resampling_scheme.lower() == 'kl':
                new_ancestors = resampling(log_joint[:, t - 1])
            elif resampling_scheme.lower() == 'kl-iw':
                new_ancestors = resampling(log_weights[:, t - 1])
            else:
                new_ancestors = resampling(normalized_weights[:, t - 1]).astype(int)
            unique_idx, multiplicities = np.unique(new_ancestors, return_counts=True)
            elbo += (multiplicities / N) @ (log_weights[unique_idx, t - 1] - np.log(N) - np.log(multiplicities / N))

            resampled_measure = np.zeros(N)
            resampled_measure[unique_idx] = multiplicities / N
            tvs[t] = 0.5 * (np.sum(
                np.abs(resampled_measure[unique_idx] - normalized_weights[unique_idx, t-1]))
                            + 1 - normalized_weights[unique_idx, t-1].sum())
            normalized_weights[:, t - 1] = 1 / N
            log_weights[:, t - 1] = 0
        else:
            new_ancestors = list(range(N))

        # == Propagate == #
        B[:, t] = new_ancestors
        particles[:, :, t] = model.propagate(particles[new_ancestors, :, t - 1])

        # == Compute weights == #
        log_g_t = model.log_g(particles[:, :d_y, t], y[t])  # incremental weight function
        normalized_weights[:, t], log_weights[:, t] = update_weights(log_weights[:, t - 1], log_g_t)

        # == Update data log-likelihood == #
        log_l_data[:, t] = log_g_t + log_l_data[new_ancestors, t - 1]

        # == Update log prior == #
        log_f_t = model.log_f(particles[..., t], particles[new_ancestors, :, t - 1])
        log_l_latent[:, t] = log_f_t + log_l_latent[new_ancestors, t - 1]

        # == Update joint likelihood == #
        log_joint[:, t] = log_g_t + log_f_t + log_joint[new_ancestors, t - 1]

        # == Marg. Log-Likelihood == #
        marg_log_likelihood += logsumexp(log_weights[:, t] - np.log(N))

    ESS[-1] = 1 / np.sum(normalized_weights[:, T - 1] ** 2)

    # == Sample sequence from resulting posterior == #
    B = B.astype(int)
    b = np.where(np.random.uniform(size=1) < np.cumsum(normalized_weights[:, T - 1]))[0][0]
    x_star = np.zeros((d, T))
    log_likelihood = np.zeros(T)
    indx = b
    for t in reversed(range(T)):
        x_star[:, t] = particles[indx, :, t]  # sampled particle trajectory
        log_likelihood[t] = log_joint[indx, t]  # joint likelihood of sampled trajectory
        indx = B[indx, t]

    out = {'x_star': x_star, 'posterior': normalized_weights[:, -1], 'ESS': ESS,
           'elbo': elbo, 'particles': particles, 'B': B,
           'll': log_likelihood, 'logjoint': log_joint, 'data_ll': log_l_data, 'latent_ll': log_l_latent,
           'marg_log_likelihood': marg_log_likelihood, 'tvs': tvs}
    return out


def update_weights(log_weights, log_g_t):
    log_weights += log_g_t
    log_w_tilde = log_weights - logsumexp(log_weights)
    normalized_weights = np.exp(log_w_tilde)
    return normalized_weights, log_weights


def resample_criterion(adaptive, ESS, N):
    if adaptive:
        return ESS < N / 2
    else:
        return True


