import numpy as np
import heapq


def multinomial_resampling(ws, size=0):
    u = np.random.rand(*ws.shape)
    bins = np.cumsum(ws)
    return np.digitize(u, bins)


def stratified_resampling(ws, size=0):
    # Determine number of elements
    N = len(ws)
    u = (np.arange(N) + np.random.rand(N)) / N
    bins = np.cumsum(ws)
    return np.digitize(u, bins)


def systematic_resampling(ws):
    N = len(ws)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    ind = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(ws)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            ind[i] = j
            i += 1
        else:
            j += 1
    return ind


def kl(log_joint, return_multiplicities=False):
    N = len(log_joint)
    # inputted, unsorted log-likelihood
    logL_s = log_joint

    # init. multiplicity, particle indices, and f_1, f_0
    B_s = np.zeros(N, dtype=np.int32)
    ss = np.arange(0, N)
    f_add = (B_s + 1) * (logL_s - np.log(B_s + 1))
    f_same = np.zeros_like(logL_s)

    # populate heap (negate f_add in order to create max heap)
    heap = []
    for c, idx in zip(f_add, ss):
        heapq.heappush(heap, (-c, idx))
    heap = heapsort(heap)

    # distribute multiplicity
    while np.sum(B_s) < N:
        C_add_max, t_add = heapq.heappop(heap)
        f_same[t_add] = f_add[t_add]
        B_s[t_add] += 1
        f_add[t_add] = (B_s[t_add] + 1) * (logL_s[t_add] - np.log(B_s[t_add] + 1))
        C_add_new = f_add[t_add] - f_same[t_add]
        heapq.heappush(heap, (-C_add_new, t_add))
    # newAncestors = np.zeros(N, dtype=np.int32)
    # idx = np.arange(N)
    # for s, b_s in enumerate(B_s):
    #     newAncestors[idx[:b_s]] = s
    #     idx = idx[b_s:]
    # ChatGPT-recommended speed up below
    newAncestors = np.repeat(np.arange(len(B_s)), B_s)

    if return_multiplicities:
        return newAncestors, B_s
    else:
        return newAncestors


def greedy_alg_beta(log_p, beta=1.0):
    N = len(log_p)
    # inputted, unsorted log-likelihood

    # init. multiplicity, particle indices, and f_1/f_0
    B_s = np.zeros(N, dtype=np.int32)
    ss = np.arange(0, N)
    f_add = (B_s + 1) * (log_p - beta * np.log(B_s + 1))
    f_same = np.zeros_like(log_p)

    # populate heap
    heap = []
    for c, idx in zip(f_add, ss):
        heapq.heappush(heap, (-c, idx))
    heap = heapsort(heap)

    # distribute multiplicity
    while np.sum(B_s) < N:
        C_add_max, t_add = heapq.heappop(heap)
        f_same[t_add] = f_add[t_add]
        B_s[t_add] += 1
        f_add[t_add] = (B_s[t_add] + 1) * (log_p[t_add] - beta * np.log(B_s[t_add] + 1))
        C_add_new = f_add[t_add] - f_same[t_add]
        heapq.heappush(heap, (-C_add_new, t_add))
    newAncestors = np.zeros(N, dtype=np.int32)
    idx = np.arange(N)
    for s, b_s in enumerate(B_s):
        newAncestors[idx[:b_s]] = s
        idx = idx[b_s:]

    return newAncestors


def heapsort(heap):
    return [heapq.heappop(heap) for _ in range(len(heap))]


def cubo(log_joint, return_multiplicities=False):
    N = len(log_joint)
    # inputted, unsorted log-likelihood
    logL_s = 2 * log_joint

    # init. multiplicity, particle indices, and f_1, f_0
    B_s = np.zeros(N, dtype=np.int32)
    ss = np.arange(0, N)
    f_add = (B_s + 1) * np.exp(2 * np.log(B_s + 1) - 2 * logL_s)
    f_same = np.zeros_like(logL_s)

    # populate heap (negate f_add in order to create max heap)
    heap = []
    for c, idx in zip(f_add, ss):
        heapq.heappush(heap, (-c, idx))
    heap = heapsort(heap)

    # distribute multiplicity
    while np.sum(B_s) < N:
        C_add_max, t_add = heapq.heappop(heap)
        f_same[t_add] = f_add[t_add]
        B_s[t_add] += 1
        f_add[t_add] = (B_s[t_add] + 1) * np.exp(2 * np.log(B_s[t_add] + 1) - 2 * logL_s[t_add])
        C_add_new = f_add[t_add] - f_same[t_add]
        heapq.heappush(heap, (-C_add_new, t_add))
    newAncestors = np.zeros(N, dtype=np.int32)
    idx = np.arange(N)
    for s, b_s in enumerate(B_s):
        newAncestors[idx[:b_s]] = s
        idx = idx[b_s:]

    if return_multiplicities:
        return newAncestors, B_s
    else:
        return newAncestors


def heapsort(heap):
    return [heapq.heappop(heap) for _ in range(len(heap))]


def ml(log_joint):
    mle = np.argmax(log_joint)
    new_ancestors = np.zeros_like(log_joint, dtype=int)
    new_ancestors[:] = mle
    return new_ancestors


def tv(ws, S=None):
    if not S:
        S = len(ws)

    u = ws * S
    floor_u = np.floor(u).astype(np.int64)
    alpha_s = u - floor_u
    alpha = S - np.sum(floor_u)
    ordered_idx = np.argsort(alpha_s)[::-1]

    B_s = np.zeros(S, dtype=np.int64)
    for i, s in enumerate(ordered_idx):
        # less than due to zero indexing
        if i < alpha:
            B_s[s] = np.ceil(u[s])
        else:
            B_s[s] = np.floor(u[s])
    newAncestors = np.zeros(S, dtype=np.int64)
    # random distribution
    idx = np.arange(S)
    for s, b_s in enumerate(B_s):
        newAncestors[idx[:b_s]] = s
        idx = idx[b_s:]
    return newAncestors


if __name__ == '__main__':
    w = np.array([0.1, 0.8, 0.05, 0.05])
    log_joint = np.array([1., 0.01, -15, -15])
    print(kl(log_joint, True))
