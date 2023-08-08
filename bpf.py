import matplotlib.pyplot as plt

from resampling import multinomial_resampling, stratified_resampling, kl, systematic_resampling, tv, cubo, greedy_alg_beta
import numpy as np
from scipy.special import logsumexp


def run_bpf(y, N, model, resampling_scheme='multinomial', adaptive=False, beta=1.):
    # Bootstrap Particle Filter (BPF)
    T = len(y)

    particles = np.zeros((N, T))
    normalized_weights = np.zeros((N, T))
    B = np.zeros((N, T))
    ESS = np.zeros(T)
    log_joint = np.zeros((N, T))
    log_weights = np.zeros((N, T))
    log_l_data = np.zeros((N, T))
    log_l_latent = np.zeros((N, T))
    predictions = np.zeros((2, T))  # 1st row mean of part. distribution, 2nd std
    mle_prediction = np.zeros(T)  # perform prediction using the maximum likelihood estimate
    filtering = np.zeros((2, T))
    mle_filtering = np.zeros(T)
    marg_log_likelihood = 0

    if resampling_scheme.lower() in 'multinomial':
        resampling = multinomial_resampling
    elif resampling_scheme.lower() in 'stratified':
        resampling = stratified_resampling
    elif resampling_scheme.lower() in 'systematic':
        resampling = systematic_resampling
    elif resampling_scheme.lower() in 'tv':
        resampling = tv
    elif resampling_scheme.lower() in 'beta':
        resampling = greedy_alg_beta
    else:
        resampling = kl

    particles[:, 0] = model.particle_0(N)
    log_g_t = model.log_g(x=particles[:, 0], y=y[0])
    log_l_data[:, 0] = log_g_t
    log_f_t = model.log_f(particles[:, 0], None)
    log_l_latent[:, 0] = log_f_t
    log_joint[:, 0] = log_l_data[:, 0] + log_l_latent[:, 0]
    normalized_weights[:, 0], log_weights[:, 0] = update_weights(log_weights=log_weights[:, 0], log_g_t=log_g_t)
    new_ancestors = list(range(N))
    B[:, 0] = new_ancestors

    betas = []

    # == Stats == #
    mle_idx = np.argmax(log_joint[:, 0])
    filtering[0, 0] = normalized_weights[:, 0] @ particles[:, 0]
    filtering[1, 0] = np.std(particles[:, 0])
    mle_filtering[0] = particles[mle_idx, 0]
    marg_log_likelihood += logsumexp(log_weights[:, 0] - np.log(N))
    predictions[0, 0] = np.mean(particles[:, 0])
    elbo = 0  # np.mean(log_weights[:, 0])

    for t in range(1, T):
        # == Resampling == #
        ESS[t - 1] = 1 / np.sum(normalized_weights[:, t - 1] ** 2)
        if resample_criterion(adaptive, ESS[t - 1], N):
            if (resampling_scheme.lower() == 'kl') or (resampling_scheme.lower() == 'cubo'):
                # new_ancestors = resampling(log_weights[:, t - 1])
                new_ancestors = resampling(log_joint[:, t - 1])
                # new_ancestors = resampling(log_g_t + log_f_t)
            elif resampling_scheme.lower() == 'beta':
                b = beta
                # if t > 3.9 * T // 4:
                #     b = beta - (t / (T-1)) ** 1 * beta + 1
                # b = np.maximum(beta * np.cos(t), 2.5)
                betas.append(b)

                # new_ancestors = resampling(log_weights[:, t - 1], beta=b)
                new_ancestors = resampling(log_weights[:, t - 1], beta=b)
            else:
                new_ancestors = resampling(normalized_weights[:, t - 1]).astype(int)
            unique_idx, multiplicities = np.unique(new_ancestors, return_counts=True)
            elbo += (multiplicities / N) @ (log_weights[unique_idx, t - 1] - np.log(N) - np.log(multiplicities / N))
            normalized_weights[:, t - 1] = 1 / N
            log_weights[:, t - 1] = 0
        else:
            new_ancestors = list(range(N))

        # filtering[0, t - 1] = (multiplicities / N) @ particles[unique_idx, t - 1]

        # == Prediction == #
        # x_preds = model.propagate(particles[unique_idx, t - 1])
        # predictions[0, t] = (multiplicities / N) @ x_preds
        # predictions[0, t] = (multiplicities / N) @ (model.phi * particles[unique_idx, t - 1])
        # predictions[0, t] = normalized_weights[:, t - 1] @ (model.phi * particles[:, t - 1])


        # == Propagate == #
        B[:, t] = new_ancestors
        particles[:, t] = model.propagate(particles[new_ancestors, t - 1])

        predictions[0, t] = np.mean(particles[:, t])
        predictions[1, t] = np.std(particles[:, t])

        # == Compute weights == #
        log_g_t = model.log_g(particles[:, t], y[t])  # incremental weight function
        normalized_weights[:, t], log_weights[:, t] = update_weights(log_weights[:, t - 1], log_g_t)

        # == Update data log-likelihood == #
        log_l_data[:, t] = log_g_t + log_l_data[new_ancestors, t - 1]

        # == Update log prior == #
        log_f_t = model.log_f(particles[:, t], particles[new_ancestors, t - 1])
        log_l_latent[:, t] = log_f_t + log_l_latent[new_ancestors, t - 1]

        # == Update joint likelihood == #
        log_joint[:, t] = log_g_t + log_f_t + log_joint[new_ancestors, t - 1]

        # == Filtering == #
        filtering[0, t] = normalized_weights[:, t] @ particles[:, t]
        # filtering[0, t - 1] = (multiplicities / N) @ particles[unique_idx, t - 1]

        # == Marg. Log-Likelihood == #
        if t < T - 1:
            marg_log_likelihood += logsumexp(log_weights[:, t] - np.log(N))

    ESS[-1] = 1 / np.sum(normalized_weights[:, T - 1] ** 2)

    # if resampling_scheme.lower() in 'beta':
    #     plt.plot(betas)
    #     plt.show()

    # == Sample sequence from resulting posterior == #
    B = B.astype(int)
    b = np.where(np.random.uniform(size=1) < np.cumsum(normalized_weights[:, T - 1]))[0][0]
    x_star = np.zeros(T)
    log_likelihood = np.zeros(T)
    indx = b
    for t in reversed(range(T)):
        x_star[t] = particles[indx, t]  # sampled particle trajectory
        log_likelihood[t] = log_joint[indx, t]  # joint likelihood of sampled trajectory
        indx = B[indx, t]

    out = {'x_star': x_star, 'posterior': normalized_weights[:, -1], 'ESS': ESS,
           'elbo': elbo, 'particles': particles, 'B': B,
           'll': log_likelihood, 'logjoint': log_joint, 'data_ll': log_l_data, 'latent_ll': log_l_latent,
           'predictions': predictions, 'mle_prediction': mle_prediction, 'filtering': filtering,
           'mle_filtering': mle_filtering, 'marg_log_likelihood': marg_log_likelihood}
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

