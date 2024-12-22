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

# +
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

sns.set()
# -

# Probability distribution
N = 1000000
a = 4/5
p = 1 / (np.arange(N) + 1) ** a
p = p / np.sum(p) # Normalize
p = p[::-1] # Reverse for convenience


# +
# Generate Buckets
def expo_buckets(min_freq, k):
    step_size = min_freq / k
    buckets = np.arange(k) * -step_size
    buckets = 10 ** buckets
    buckets = np.append(buckets, 0)
    return buckets[::-1]

def linear_buckets(k):
    return np.arange(k + 1) / (k+1)

def quantile_buckets(k):
    last_i = 0
    cum_p = np.cumsum(p)
    def get_quantile(q, last_i):
        # Find first i such that sum[:i+1] >= q
        for i in range(last_i, len(cum_p)):
            if cum_p[i] >= q:
                return i, p[i]

    buckets = np.empty(k + 1)
    buckets[0] = 0
    buckets[-1] = 1
    # k-1 quantiles, last bucket is catch-all
    for i in range(k-1):
        last_i, buckets[i+1] = get_quantile((i+1)/(k-1), last_i) 
    return buckets


# +
e_buckets = expo_buckets(7, 16)
l_buckets = linear_buckets(16)
q_buckets = quantile_buckets(16)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 5))

for i in range(16):
    ax[0].axhline(e_buckets[i], c='r', linestyle='--')
    ax[1].axhline(l_buckets[i], c='g', linestyle='--')
    ax[2].axhline(q_buckets[i], c='b', linestyle='--')

ax[0].plot(np.arange(N), p[::-1])
ax[1].plot(np.arange(N), p[::-1])
ax[2].plot(np.arange(N), p[::-1])
plt.xscale('log')
plt.yscale('log')


# +
# Estimator coefficients
def calc_unbiased_c():
    return (N-1) * (N-2) * p**2 + 3*(N-1) * p + 1

def calc_bucketing_c(buckets):
    # Return index i s.t. buckets[i] < p <= buckets[i+1]
    last_i = 0
    def get_bucket(p, last_i):
        for i in range(last_i, len(buckets) - 1):
            if buckets[i] < p <= buckets[i+1]:
                return i, buckets[i], buckets[i+1]

    c = np.empty(len(p))
    for i, p_i in enumerate(p):
        last_i, p_a, p_b = get_bucket(p_i, last_i)
        c[i] = (N * (p_a + p_b) / 2) ** 2
    return c


# +
def calc_bias(c, p):
    return np.sum(N * c * p) - np.sum(N*(N-1)*(N-2)*p**3 + 3*N*(N-1)*p**2 + N*p)

print(calc_bias(calc_unbiased_c(), p))
print(calc_bias(calc_bucketing_c(expo_buckets(7, 16)), p))
print(calc_bias(calc_bucketing_c(linear_buckets(16)), p))
print(calc_bias(calc_bucketing_c(quantile_buckets(16)), p))


# +
# Variance of estimator
def calc_var(c, p):
    return N * (np.sum(c**2 * p) - (np.sum(c * p))**2)

print(calc_var(calc_unbiased_c(), p))
print(calc_var(calc_bucketing_c(expo_buckets(7, 16)), p))
print(calc_var(calc_bucketing_c(linear_buckets(16)), p))
print(calc_var(calc_bucketing_c(quantile_buckets(16)), p))

# +
bucket_sizes = [(i+1) * 1000 for i in range(20)]

unbiased_bias, unbiased_vars = np.zeros(len(bucket_sizes)), np.empty(len(bucket_sizes))
expo_bias, expo_vars = np.empty(len(bucket_sizes)), np.empty(len(bucket_sizes))
lin_bias, lin_vars = np.empty(len(bucket_sizes)), np.empty(len(bucket_sizes))
quan_bias, quan_vars = np.empty(len(bucket_sizes)), np.empty(len(bucket_sizes))

for i, B in enumerate(tqdm(bucket_sizes)):
    cap = 2**10# int(B/2)
    
    unbiased_vars[i] = calc_var(calc_unbiased_c()[:-cap], p[:-cap])
    
    expo_c = calc_bucketing_c(expo_buckets(7, B))
    expo_bias[i], expo_vars[i] = calc_bias(expo_c[:-cap], p[:-cap]), calc_var(expo_c[:-cap], p[:-cap])

    lin_c = calc_bucketing_c(linear_buckets(B))
    lin_bias[i], lin_vars[i] = calc_bias(lin_c[:-cap], p[:-cap]), calc_var(lin_c[:-cap], p[:-cap])

    quan_c = calc_bucketing_c(quantile_buckets(B))
    quan_bias[i], quan_vars[i] = calc_bias(quan_c[:-cap], p[:-cap]), calc_var(quan_c[:-cap], p[:-cap])

# +
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(12, 6))
ax[0].plot(bucket_sizes, expo_bias)
ax[1].plot(bucket_sizes, lin_bias)
# ax[1].plot(bucket_sizes, quan_bias)
ax[2].plot(bucket_sizes, unbiased_bias)

ax[0].set_title("Exponential buckets")
ax[1].set_title("Linear buckets")
ax[2].set_title("Unbiased estimator")

# ax[0].set_xticks(bucket_sizes, bucket_sizes, rotation='vertical')
# ax[1].set_xticks(bucket_sizes, bucket_sizes, rotation='vertical')
# ax[2].set_xticks(bucket_sizes, bucket_sizes, rotation='vertical')

fig.supxlabel("Number of Buckets $B$")
fig.supylabel("Bias")
fig.suptitle("Bias vs. number of buckets")

# plt.savefig('figs/tophh_bias_graph')
plt.show()

# +
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 6))
ax[0].plot(bucket_sizes, expo_vars)
ax[1].plot(bucket_sizes, lin_vars)
# ax[1].plot(bucket_sizes, quan_vars)
ax[2].plot(bucket_sizes, unbiased_vars)

ax[0].set_title("Exponential buckets")
ax[1].set_title("Linear buckets")
ax[2].set_title("Unbiased estimator")

# plt.xscale('log')
plt.yscale('log')

# ax[0].set_xticks(bucket_sizes, bucket_sizes, rotation='vertical')
# ax[1].set_xticks(bucket_sizes, bucket_sizes, rotation='vertical')
# ax[2].set_xticks(bucket_sizes, bucket_sizes, rotation='vertical')

fig.supxlabel("Number of Buckets $B$")
fig.supylabel("Variance")
fig.suptitle("Variance vs. number of buckets")

# plt.savefig('figs/tophh_variance_graph')
plt.show()
