# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# +
# Work with perfectly uniform bucket at first
n = 10000 # Number of unique elements
N = 100000 # Total sum of elements
d = 3 # d-th moment

rng = np.random.default_rng()

def sim_uniform_sample(print_table=False):
    els = rng.multinomial(N, np.ones(n) / n)
    
    # Try different number to sample
    true_val = np.sum(els ** d)
    t = PrettyTable(["k", "est", "error"])
    
    ks = np.arange(10, n, 10)
    errs = []
    for k in ks:
        sample = els[:k]
        est = np.sum(sample ** d) * N / np.sum(sample)
        err = np.abs(est - true_val) / true_val
        errs.append(err)
        t.add_row([k, est, err])
    
    if print_table: print(t)
    plt.plot(ks, errs)

    est = (N / n)**d * n
    plt.axhline(np.abs(est - true_val) / true_val)

for _ in range(10):
    sim_uniform_sample()

# plt.yscale('log')
plt.xscale('log')
plt.show()

# +
# Model a single bucket
lower_rank = 50
upper_rank = 100
N = 10000
alpha = 2

p = 1 / (np.arange(upper_rank, step=1) + 1) ** alpha
p = p / np.sum(p)

# +
rng = np.random.default_rng()
def simulate(N, p):
    X = rng.multinomial(N, p / np.sum(p))
    p = p[lower_rank - 1:]
    X = X[lower_rank - 1:]

    # Exact moment
    exact_moment = np.sum(X**3)

    # Distribution estimator
    S = np.sum(X)
    p_sum = (upper_rank ** (alpha - 1) - lower_rank ** (alpha - 1)) / ((upper_rank * lower_rank) ** (alpha - 1) * (alpha - 1))
    p2_sum = (upper_rank ** (2*alpha - 1) - lower_rank ** (2*alpha - 1)) / ((upper_rank * lower_rank) ** (2*alpha - 1) * (2*alpha - 1))
    p3_sum = (upper_rank ** (3*alpha - 1) - lower_rank ** (3*alpha - 1)) / ((upper_rank * lower_rank) ** (3*alpha - 1) * (3*alpha - 1))
    dist_est = S * (S-1) * (S-2) * p3_sum + 3 * S * (S-1) * p2_sum + S * p_sum

    # Conditional estimator
    Sp = np.sum(p * X)
    Sp2 = np.sum(p**2 * X)
    corr_factor = N / S
    cond_est = (S-1) * (S-2) * Sp2 * corr_factor**2 + 3 * (S-1) * Sp * corr_factor + S

    print(N / S, 1 / np.sum(p))

    # Conditional estimator normalized
    p_norm = p / np.sum(p)
    Sp = np.sum(p_norm * X)
    Sp2 = np.sum(p_norm**2 * X)
    cond_est_norm = (S-1) * (S-2) * Sp2 + 3 * (S-1) * Sp + S

    return (exact_moment, dist_est, cond_est, cond_est_norm)

nsims = 30
exact_moments = np.empty(nsims)
dist_ests = np.empty(nsims)
cond_ests = np.empty(nsims)
cond_ests_norm = np.empty(nsims)

for i in range(nsims):
    exact_moments[i], dist_ests[i], cond_ests[i], cond_ests_norm[i] = simulate(N, p)


# Expected moment
# p_norm = p[lower_rank - 1:] / np.sum(p[lower_rank - 1:])
# expected_moment = N * (N-1) * (N-2) * np.sum(p_norm**3) + 3 * N * (N-1) * np.sum(p_norm**2) + N

print(exact_moments)
print(dist_ests)
print(cond_ests)
print(cond_ests_norm)

t = PrettyTable(["Type", "Avg", "SD", "NRMSE"])
def NRMSE(est):
    return np.sqrt(np.mean((exact_moments - est)**2))
t.add_row(["Exact", np.mean(exact_moments), np.std(exact_moments), NRMSE(exact_moments)])
t.add_row(["Dist", np.mean(dist_ests), np.std(dist_ests), NRMSE(dist_ests)])
t.add_row(["Cond", np.mean(cond_ests), np.std(cond_ests), NRMSE(cond_ests)])
t.add_row(["Cond Norm", np.mean(cond_ests_norm), np.std(cond_ests_norm), NRMSE(cond_ests_norm)])
print(expected_moment)
print(t)

