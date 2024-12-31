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
import pandas as pd
from tqdm.notebook import tqdm, trange
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# +
### Load data
df1 = pd.read_csv('data/AOL-user-ct-collection/user-ct-test-collection-01.txt', sep='\t')
df1['Query'] = df1['Query'].fillna("")
unique_counts1 = df1['Query'].value_counts()
unique_freqs1 = unique_counts1 / len(df1)

df2 = pd.read_csv('data/AOL-user-ct-collection/user-ct-test-collection-02.txt', sep='\t')
df2['Query'] = df2['Query'].fillna("")
unique_counts2 = df2['Query'].value_counts()
unique_freqs2 = unique_counts2 / len(df2)
# -

# print(df.head())
# print(unique_counts.head(), len(unique_counts))
# print(unique_freqs.head())
print(unique_counts1[~unique_counts1.index.isin(unique_counts2.index)])

# +
for k in range(16):
    # plt.axhline(10 ** (-7*k/16), c='r', linestyle='--')
    plt.axhline(k/2048, c='g', linestyle='--')

plt.plot(np.arange(len(unique_counts1)), unique_freqs1, c='b')
plt.plot(np.arange(len(unique_counts2)), unique_freqs2, c='r')


plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Rank vs. Frequency of search results')


# -

# Generate Buckets
def generate_buckets(min_freq, k):
    step_size = min_freq / k
    buckets = np.arange(k) * -step_size
    buckets = 10 ** buckets
    buckets = np.append(buckets, 0)
    return buckets[::-1]


# +
# Plot buckets
def plot_buckets(buckets):
    for bucket in buckets:
        plt.axhline(bucket, c='r')
    plt.plot(np.arange(len(unique_counts)), unique_freqs, c='b')
    
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Rank vs. Frequency of search results')

plot_buckets(generate_buckets(7, 32))


# +
# Place each item into a bucket
def find_bucket_index(buckets, value):
    return np.searchsorted(buckets, value, side='left') - 1

def place_in_buckets(buckets, est_freqs):
    k = len(buckets) - 1
    filled_buckets = [[] for _ in range(k)]
    for item, freq in est_freqs.items():
        filled_buckets[find_bucket_index(buckets, freq)].append(unique_counts[item])
    for i in range(k):
        filled_buckets[i] = np.array(filled_buckets[i])
    return filled_buckets


# -

# Helper Functions
def norm(items, p):
    if len(items) == 0:
        return np.nan
    
    return np.sum(items ** p)


# +
p = 3
t = PrettyTable(["Left", "Right", "N", "S", "||x||^p", "n*||x/n||^p"])

k = 256
buckets = generate_buckets(7, k)
filled_buckets = place_in_buckets(buckets, unique_freqs)

bucket_norms = np.empty(k)
bucket_preds = np.empty(k)

for i, items in enumerate(filled_buckets):
    left, right = buckets[i] * len(df), buckets[i+1] * len(df)
    n = len(items)
    S = items.sum()
    
    nhat = np.round(S / np.sqrt(left * right)) if S > 0 else np.nan
    pred = (S**p)/(nhat**(p-1)) if S > 0 else np.nan

    bucket_norms[i] = norm(items, p)
    bucket_preds[i] = pred
    
    t.add_row([left, right, n, S, norm(items, p), pred])

print(np.nan_to_num(bucket_norms).sum(), np.nan_to_num(bucket_preds).sum())
print(t)

# +
# Plot loss for each bucket type
bucket_errors_abs = bucket_norms - bucket_preds
bucket_errors_rel = bucket_errors_abs / bucket_norms

plt.scatter(np.arange(k)[bucket_errors_abs != np.nan], bucket_errors_abs[bucket_errors_abs != np.nan])
plt.show()

plt.scatter(np.arange(k)[bucket_errors_rel != np.nan], bucket_errors_rel[bucket_errors_rel != np.nan])
plt.show()


# -

def calculate_errors(loss_func, ks, p, min_freq = 7):    
    errors = np.empty(len(ks))
    for i, k in enumerate(tqdm(ks)):
        bucket_norms, bucket_preds = loss_func(k, p, min_freq)
        total_norms, total_preds = np.nan_to_num(bucket_norms).sum(), np.nan_to_num(bucket_preds).sum()
        errors[i] = (total_norms - total_preds) / total_norms
    return errors


# +
def simulate_loss_exact(k, p, min_freq):
    buckets = generate_buckets(min_freq, k)
    filled_buckets = place_in_buckets(buckets, unique_freqs)
    
    bucket_norms = np.empty(k)
    bucket_preds = np.empty(k)
    
    for i, items in enumerate(filled_buckets):
        left, right = buckets[i], buckets[i+1]
        S = items.sum()
        
        n = len(items)
        pred = (S**p)/(n**(p-1)) if n > 0 else np.nan
    
        bucket_norms[i] = norm(items, p)
        bucket_preds[i] = pred

    return bucket_norms, bucket_preds

# Check errors when we don't have N (left and right)
# Know S / right <= N <= S / left

def simulate_loss_estimate(est_freqs, n_estimate, k, p, min_freq):
    buckets = generate_buckets(min_freq, k)
    filled_buckets = place_in_buckets(buckets, est_freqs)
    
    bucket_norms = np.empty(k)
    bucket_preds = np.empty(k)
    
    for i, items in enumerate(filled_buckets):
        left, right = buckets[i] * len(df), buckets[i+1] * len(df)
        S = items.sum()

        n = n_estimate(S, left, right) if S > 0 else np.nan
        n = np.maximum(1, n)
        pred = (S**p)/(n**(p-1)) if S > 0 else np.nan
    
        bucket_norms[i] = norm(items, p)
        bucket_preds[i] = pred

    return bucket_norms, bucket_preds

def n_estimate_left(S, left, right):
    return np.ceil(S / left)

def n_estimate_right(S, left, right):
    return np.floor(S / right)

def n_estimate_left_round(S, left, right):
    return np.ceil(S / np.ceil(left))

def n_estimate_right_round(S, left, right):
    return np.floor(S / np.floor(right))

def n_estimate_arith_avg(S, left, right):
    return np.round(S * (left + right)/(2 * left * right))

def n_estimate_geo_avg(S, left, right):
    return np.round(S / np.sqrt(left * right))

def n_estimate_harm_avg(S, left, right):
    return np.round(2 * S / (left + right))


# +
rng = np.random.default_rng()

def est_rel_freqs(ep = 0.1):
    est_freqs = unique_freqs.copy()
    err = rng.uniform(1-ep, 1+ep, len(est_freqs))

    i = 0
    for item, freq in unique_freqs.items():
        est_freqs[item] = freq * err[i]
        i += 1
    return np.clip(est_freqs, 1/len(df), 1)

def est_abs_freqs(ep = 0.1):
    est_freqs = unique_freqs.copy()
    err = rng.uniform(-ep, +ep, len(est_freqs))

    i = 0
    for item, freq in unique_freqs.items():
        est_freqs[item] = freq + err[i]
        i += 1
    return np.clip(est_freqs, 1/len(df), 1)

def est_bin_freqs():
    est_freqs = unique_freqs.copy()

    for item, freq in unique_freqs.items():
        est_freqs[item] = rng.binomial(len(df), freq) / len(df)
    return np.clip(est_freqs, 1/len(df), 1)


# +
p = 3
t = PrettyTable(["Left", "Right", "S", "N", "N_left", "N_right", "N_left_round", "N_right_round", "N_arith_avg", "N_geo_avg", "N_harm_avg"])

k = 32768
buckets = generate_buckets(7, k)
est_freqs = est_rel_freqs(0.2)
filled_buckets = place_in_buckets(buckets, est_freqs)

for i, items in enumerate(filled_buckets):
    left, right = buckets[i] * len(df), buckets[i+1] * len(df)
    n = len(items)
    if n == 0:
        continue
    
    S = items.sum()
    n_left = n_estimate_left(S, left, right) if S > 0 else np.nan
    n_right = n_estimate_right(S, left, right) if S > 0 else np.nan
    n_left_round = n_estimate_left_round(S, left, right) if S > 0 else np.nan
    n_right_round = n_estimate_right_round(S, left, right) if S > 0 else np.nan
    n_arith_avg = n_estimate_arith_avg(S, left, right) if S > 0 else np.nan
    n_geo_avg = n_estimate_geo_avg(S, left, right) if S > 0 else np.nan
    n_harm_avg = n_estimate_harm_avg(S, left, right) if S > 0 else np.nan

    n_left = np.maximum(1, n_left)
    n_right = np.maximum(1, n_right)
    n_left_round = np.maximum(1, n_left_round)
    n_right_round = np.maximum(1, n_right_round)
    n_arith_avg = np.maximum(1, n_arith_avg)
    n_geo_avg = np.maximum(1, n_geo_avg)
    n_harm_avg = np.maximum(1, n_harm_avg)

    t.add_row([left, right, S, n, n_left, n_right, n_left_round, n_right_round, n_arith_avg, n_geo_avg, n_harm_avg])

print(t)

# + jupyter={"outputs_hidden": true}
est_freqs = est_rel_freqs(0.05)
def get_simulate_loss_estimate(n_estimate):
    return lambda k, p, min_freq: simulate_loss_estimate(est_freqs, n_estimate, k, p, min_freq)

ks = [2**i for i in range(8, 17)]
p, min_freq = 3, 7

errors = [
    calculate_errors(simulate_loss_exact, ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_left), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_right), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_left_round), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_right_round), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_arith_avg), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_geo_avg), ks, p, min_freq),
    calculate_errors(get_simulate_loss_estimate(n_estimate_harm_avg), ks, p, min_freq),
]
# "left", "right", "left_round", "right_round", "arith mean", "geo mean",
t = PrettyTable(["k", "exact", "harm mean"])
for row in zip(ks, *errors):
    best = min([abs(x) for x in row[2:]])
    t.add_row([row[0]] + [f"**{x:.3e}**" if abs(x) == best else f"{x:.3e}" for x in row[1:]])
print(t)

# +
est_freqs = est_abs_freqs(0.05)
def get_simulate_loss_estimate(n_estimate):
    return lambda k, p, min_freq: simulate_loss_estimate(est_freqs, n_estimate, k, p, min_freq)

ks = [2**i for i in range(8, 17)]
p, min_freq = 3, 7

errors = [
    calculate_errors(simulate_loss_exact, ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_left), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_right), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_left_round), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_right_round), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_arith_avg), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_geo_avg), ks, p, min_freq),
    calculate_errors(get_simulate_loss_estimate(n_estimate_harm_avg), ks, p, min_freq),
]
# "left", "right", "left_round", "right_round", "arith mean", "geo mean",
t = PrettyTable(["k", "exact", "harm mean"])
for row in zip(ks, *errors):
    best = min([abs(x) for x in row[2:]])
    t.add_row([row[0]] + [f"**{x:.3e}**" if abs(x) == best else f"{x:.3e}" for x in row[1:]])
print(t)

# +
est_freqs = est_bin_freqs()
def get_simulate_loss_estimate(n_estimate):
    return lambda k, p, min_freq: simulate_loss_estimate(est_freqs, n_estimate, k, p, min_freq)

ks = [2**i for i in range(8, 17)]
p, min_freq = 3, 7

errors = [
    calculate_errors(simulate_loss_exact, ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_left), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_right), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_left_round), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_right_round), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_arith_avg), ks, p, min_freq),
    # calculate_errors(get_simulate_loss_estimate(n_estimate_geo_avg), ks, p, min_freq),
    calculate_errors(get_simulate_loss_estimate(n_estimate_harm_avg), ks, p, min_freq),
]
# "left", "right", "left_round", "right_round", "arith mean", "geo mean",
t = PrettyTable(["k", "exact", "harm mean"])
for row in zip(ks, *errors):
    best = min([abs(x) for x in row[2:]])
    t.add_row([row[0]] + [f"**{x:.3e}**" if abs(x) == best else f"{x:.3e}" for x in row[1:]])
print(t)

# +
# Try bucketing using oracle frequencies with errors
nsims = 30
ep = 0.05
sample_sizes = [2**i for i in range(8, 17)]
sim_folder = "simulation/results"
def read_sim_data(path):
    with open(f"{sim_folder}/{path}", 'r') as f:
        lines = [line.rstrip() for line in f]
        return [float.fromhex(lines[i]) for i in range(min(len(lines), nsims))]

true_value = float(sum([val**3 for val in unique_counts]))

expo_arith_abs = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=arith_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_bin = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=arith_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_rel = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=arith_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_exact = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=arith_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_geo_abs = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=geo_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_bin = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=geo_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_rel = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=geo_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_exact = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=geo_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_harm_abs = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=harm_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_bin = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=harm_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_rel = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=harm_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_exact = [read_sim_data(f"expo_bucket_k={k}_khh=0_min_freq=7_n_est=harm_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_arith_abs_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=arith_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_bin_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=arith_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_rel_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=arith_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_exact_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=arith_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_geo_abs_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=geo_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_bin_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=geo_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_rel_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=geo_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_exact_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=geo_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_harm_abs_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=harm_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_bin_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=harm_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_rel_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=harm_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_exact_hh = [read_sim_data(f"expo_bucket_k={k//2}_khh={k//2}_min_freq=7_n_est=harm_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

def plot_curve(ax, x, results, true_value, label, color):
    results = np.array(results)
    error = np.abs((results - true_value) / true_value)
    mean = np.mean(error, axis=1)
    lower = np.min(error, axis=1)
    upper = np.max(error, axis=1)
    
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    # mse = np.sqrt(np.mean((results - true_value)**2, axis=1)) / true_value    
    # ax.plot(x, mse, label=label, color=color)
    
    ax.legend()

fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(9, 9))
    
plot_curve(ax[0][0], sample_sizes, expo_arith_abs, true_value, "expo_arith_abs", 'b')
plot_curve(ax[0][1], sample_sizes, expo_arith_bin, true_value, "expo_arith_bin", 'b')
plot_curve(ax[0][2], sample_sizes, expo_arith_rel, true_value, "expo_arith_rel", 'b')
plot_curve(ax[0][3], sample_sizes, expo_arith_exact, true_value, "expo_arith_exact", 'b')

plot_curve(ax[1][0], sample_sizes, expo_geo_abs, true_value, "expo_geo_abs", 'b')
plot_curve(ax[1][1], sample_sizes, expo_geo_bin, true_value, "expo_geo_bin", 'b')
plot_curve(ax[1][2], sample_sizes, expo_geo_rel, true_value, "expo_geo_rel", 'b')
plot_curve(ax[1][3], sample_sizes, expo_geo_exact, true_value, "expo_geo_exact", 'b')

plot_curve(ax[2][0], sample_sizes, expo_harm_abs, true_value, "expo_harm_abs", 'b')
plot_curve(ax[2][1], sample_sizes, expo_harm_bin, true_value, "expo_harm_bin", 'b')
plot_curve(ax[2][2], sample_sizes, expo_harm_rel, true_value, "expo_harm_rel", 'b')
plot_curve(ax[2][3], sample_sizes, expo_harm_exact, true_value, "expo_harm_exact", 'b')

plot_curve(ax[0][0], sample_sizes, expo_arith_abs_hh, true_value, "expo_arith_abs", 'g')
plot_curve(ax[0][1], sample_sizes, expo_arith_bin_hh, true_value, "expo_arith_bin", 'g')
plot_curve(ax[0][2], sample_sizes, expo_arith_rel_hh, true_value, "expo_arith_rel", 'g')
plot_curve(ax[0][3], sample_sizes, expo_arith_exact_hh, true_value, "expo_arith_exact", 'g')

plot_curve(ax[1][0], sample_sizes, expo_geo_abs_hh, true_value, "expo_geo_abs", 'g')
plot_curve(ax[1][1], sample_sizes, expo_geo_bin_hh, true_value, "expo_geo_bin", 'g')
plot_curve(ax[1][2], sample_sizes, expo_geo_rel_hh, true_value, "expo_geo_rel", 'g')
plot_curve(ax[1][3], sample_sizes, expo_geo_exact_hh, true_value, "expo_geo_exact", 'g')

plot_curve(ax[2][0], sample_sizes, expo_harm_abs_hh, true_value, "expo_harm_abs", 'g')
plot_curve(ax[2][1], sample_sizes, expo_harm_bin_hh, true_value, "expo_harm_bin", 'g')
plot_curve(ax[2][2], sample_sizes, expo_harm_rel_hh, true_value, "expo_harm_rel", 'g')
plot_curve(ax[2][3], sample_sizes, expo_harm_exact_hh, true_value, "expo_harm_exact", 'g')

ax[0][0].set_xscale('log')
ax[0][0].set_yscale('log')

ax[2][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
ax[2][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
ax[2][2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
ax[2][3].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

fig.supxlabel('Sample size')
fig.supylabel(f'Normalized Estimation Error')
fig.suptitle(f'Bucket sketches')

# plt.savefig(f'figs/freq_sketch_sim_{deg}_moment_{prefix}_err_large_sample')
plt.show()

# +
# Try bucketing using oracle frequencies with errors
nsims = 30
sample_sizes = [2**i for i in range(8, 17)]
sim_folder = "simulation/results"
def read_sim_data(path):
    with open(f"{sim_folder}/{path}", 'r') as f:
        lines = [line.rstrip() for line in f]
        return [float.fromhex(lines[i]) for i in range(min(len(lines), nsims))]

true_value = float(sum([val**3 for val in unique_counts]))

linear_arith_abs = [read_sim_data(f"linear_bucket_k={k}_n_est=arith_abs_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_arith_bin = [read_sim_data(f"linear_bucket_k={k}_n_est=arith_bin_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_arith_rel = [read_sim_data(f"linear_bucket_k={k}_n_est=arith_rel_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_arith_exact = [read_sim_data(f"linear_bucket_k={k}_n_est=arith_exact_ep=0.05_deg=3.txt") for k in sample_sizes]

linear_geo_abs = [read_sim_data(f"linear_bucket_k={k}_n_est=geo_abs_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_geo_bin = [read_sim_data(f"linear_bucket_k={k}_n_est=geo_bin_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_geo_rel = [read_sim_data(f"linear_bucket_k={k}_n_est=geo_rel_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_geo_exact = [read_sim_data(f"linear_bucket_k={k}_n_est=geo_exact_ep=0.05_deg=3.txt") for k in sample_sizes]

linear_harm_abs = [read_sim_data(f"linear_bucket_k={k}_n_est=harm_abs_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_harm_bin = [read_sim_data(f"linear_bucket_k={k}_n_est=harm_bin_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_harm_rel = [read_sim_data(f"linear_bucket_k={k}_n_est=harm_rel_ep=0.05_deg=3.txt") for k in sample_sizes]
linear_harm_exact = [read_sim_data(f"linear_bucket_k={k}_n_est=harm_exact_ep=0.05_deg=3.txt") for k in sample_sizes]

def plot_curve(ax, x, results, true_value, label, color):
    results = np.array(results)
    # error = np.abs((results - true_value) / true_value)
    # mean = np.mean(error, axis=1)
    # lower = np.min(error, axis=1)
    # upper = np.max(error, axis=1)
    
    # ax.plot(x, mean, label=label, color=color)
    # ax.fill_between(x, lower, upper, color=color, alpha=0.2)

    mse = np.sqrt(np.mean((results - true_value)**2, axis=1)) / true_value    
    ax.plot(x, mse, label=label, color=color)
    
    ax.legend()

fig, ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(9, 9))
    
plot_curve(ax[0][0], sample_sizes, linear_arith_abs, true_value, "linear_arith_abs", 'b')
plot_curve(ax[0][1], sample_sizes, linear_arith_bin, true_value, "linear_arith_bin", 'b')
plot_curve(ax[0][2], sample_sizes, linear_arith_rel, true_value, "linear_arith_rel", 'b')
plot_curve(ax[0][3], sample_sizes, linear_arith_exact, true_value, "linear_arith_exact", 'b')

plot_curve(ax[1][0], sample_sizes, linear_geo_abs, true_value, "linear_geo_abs", 'b')
plot_curve(ax[1][1], sample_sizes, linear_geo_bin, true_value, "linear_geo_bin", 'b')
plot_curve(ax[1][2], sample_sizes, linear_geo_rel, true_value, "linear_geo_rel", 'b')
plot_curve(ax[1][3], sample_sizes, linear_geo_exact, true_value, "linear_geo_exact", 'b')

plot_curve(ax[2][0], sample_sizes, linear_harm_abs, true_value, "linear_harm_abs", 'b')
plot_curve(ax[2][1], sample_sizes, linear_harm_bin, true_value, "linear_harm_bin", 'b')
plot_curve(ax[2][2], sample_sizes, linear_harm_rel, true_value, "linear_harm_rel", 'b')
plot_curve(ax[2][3], sample_sizes, linear_harm_exact, true_value, "linear_harm_exact", 'b')

ax[0][0].set_xscale('log')
ax[0][0].set_yscale('log')

fig.supxlabel('Sample size')
fig.supylabel(f'Normalized Estimation Error')
fig.suptitle(f'Bucket sketches')

# plt.savefig(f'figs/freq_sketch_sim_{deg}_moment_{prefix}_err_large_sample')
plt.show()

# +
# Try bucketing using oracle frequencies with errors
nsims = 30
ep = 0.05
sample_sizes = [2**i for i in range(4, 17)]
sim_folder = "simulation/results"
def read_sim_data(path):
    with open(f"{sim_folder}/{path}", 'r') as f:
        lines = [line.rstrip() for line in f]
        return [float.fromhex(lines[i]) for i in range(min(len(lines), nsims))]

true_value = float(sum([val**3 for val in unique_counts]))

expo_arith_abs = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=arith_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_bin = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=arith_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_rel = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=arith_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_arith_exact = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=arith_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_geo_abs = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=geo_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_bin = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=geo_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_rel = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=geo_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_geo_exact = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=geo_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

expo_harm_abs = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=harm_abs_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_bin = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=harm_bin_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_rel = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=harm_rel_ep={ep}_deg=3.txt") for k in sample_sizes]
expo_harm_exact = [read_sim_data(f"expo_bucket_k={k}_min_freq=7_n_est=harm_exact_ep={ep}_deg=3.txt") for k in sample_sizes]

def plot_curve(ax, x, results, true_value, label, color):
    results = np.array(results)

    mse = np.sqrt(np.mean((results - true_value)**2, axis=1)) / true_value    
    ax.plot(x, mse, label=label, color=color)
    
    ax.legend()

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 9))
    
plot_curve(ax[0], sample_sizes, expo_arith_abs, true_value, "expo_arith_abs", 'r')
plot_curve(ax[0], sample_sizes, expo_arith_bin, true_value, "expo_arith_bin", 'g')
plot_curve(ax[0], sample_sizes, expo_arith_rel, true_value, "expo_arith_rel", 'b')
plot_curve(ax[0], sample_sizes, expo_arith_exact, true_value, "expo_arith_exact", 'y')

plot_curve(ax[1], sample_sizes, expo_geo_abs, true_value, "expo_geo_abs", 'r')
plot_curve(ax[1], sample_sizes, expo_geo_bin, true_value, "expo_geo_bin", 'g')
plot_curve(ax[1], sample_sizes, expo_geo_rel, true_value, "expo_geo_rel", 'b')
plot_curve(ax[1], sample_sizes, expo_geo_exact, true_value, "expo_geo_exact", 'y')

plot_curve(ax[2], sample_sizes, expo_harm_abs, true_value, "expo_harm_abs", 'r')
plot_curve(ax[2], sample_sizes, expo_harm_bin, true_value, "expo_harm_bin", 'g')
plot_curve(ax[2], sample_sizes, expo_harm_rel, true_value, "expo_harm_rel", 'b')
plot_curve(ax[2], sample_sizes, expo_harm_exact, true_value, "expo_harm_exact", 'y')

ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

fig.supxlabel('Sample size')
fig.supylabel(f'Normalized Estimation Error')
fig.suptitle(f'Bucket sketches')

# plt.savefig(f'figs/freq_sketch_sim_{deg}_moment_{prefix}_err_large_sample')
plt.show()

# +
# Plot error in counts between exact + estimated buckets
buckets = generate_buckets(7, 1024)
filled_buckets_exact = place_in_buckets(buckets, unique_freqs)
est_freqs = est_rel_freqs(0.05)
filled_buckets_est = place_in_buckets(buckets, est_freqs)

actual_preds = np.empty(len(filled_buckets_exact))
exact_preds = np.empty(len(filled_buckets_exact))
est_preds = np.empty(len(filled_buckets_est))

t = PrettyTable(["Left", "Right", "S_exact", "S_est", "actual_n", "n_exact", "n_est", "Actual norm", "Exact pred", "Est pred"])
for i, (exact_items, est_items) in enumerate(zip(filled_buckets_exact, filled_buckets_est)):
    left, right = buckets[i] * len(df), buckets[i+1] * len(df)

    S_exact = exact_items.sum()
    S_est = est_items.sum()

    n_exact = n_estimate_harm_avg(S_exact, left, right) if S_exact > 0 else np.nan
    n_exact = np.maximum(1, n_exact)
    pred_exact = (S_exact**p)/(n_exact**(p-1)) if S_exact > 0 else np.nan

    n_est = n_estimate_harm_avg(S_est, left, right) if S_est > 0 else np.nan
    n_est = np.maximum(1, n_est)
    pred_est = (S_est**p)/(n_est**(p-1)) if S_est > 0 else np.nan

    actual_n = len(exact_items)
    actual_norm = norm(exact_items, p)

    actual_preds[i] = actual_norm
    exact_preds[i] = pred_exact
    est_preds[i] = pred_est
    
    t.add_row([f"{left:.3e}", f"{right:.3e}", S_exact, S_est, actual_n, n_exact, n_est, actual_norm, pred_exact, pred_est])

actual_preds = np.nan_to_num(actual_preds)
exact_preds = np.nan_to_num(exact_preds)
est_preds = np.nan_to_num(est_preds)
# -

print(t)

# +
# Predicted norm per bucket

true_val = float(sum([val**p for val in unique_counts]))

print(true_val)
print(np.sum(exact_preds), (true_val - np.sum(exact_preds)) / true_val)
print(np.sum(est_preds), (true_val - np.sum(est_preds)) / true_val)

a = np.sum(exact_preds[400:600]) + np.sum(est_preds[:400]) + np.sum(est_preds[600:])
print(a, (true_val - a) / true_val)

plt.scatter(np.arange(len(exact_preds)), actual_preds)
plt.scatter(np.arange(len(exact_preds)), est_preds)

# plt.scatter(np.arange(len(exact_preds)), np.abs(exact_preds - est_preds))

plt.yscale('log')

# +
# Predicted norm per item
actual_item_norms = np.empty(len(unique_counts))
pred_exact_item_norms = np.empty(len(unique_counts))
pred_est_item_norms = np.empty(len(unique_counts))

for i, count in enumerate(unique_counts):
    actual_item_norms[i] = count**p

    item_bucket = find_bucket_index(buckets, unique_freqs.iloc[i])
    pred_exact_item_norms[i] = exact_preds[item_bucket] / len(filled_buckets_exact[item_bucket])

    item_bucket = find_bucket_index(buckets, est_freqs.iloc[i])
    pred_est_item_norms[i] = est_preds[item_bucket] / len(filled_buckets_est[item_bucket])
# -

t = PrettyTable(["Actual norm", "Exact norm", "Est norm"])
for i in range(100):
    t.add_row([actual_item_norms[i], pred_exact_item_norms[i], pred_est_item_norms[i]])
print(t)

# +
true_val = float(sum([val**p for val in unique_counts]))
a = np.sum(actual_item_norms[:256]) + np.sum(pred_est_item_norms[256:])
print(a, (a - true_val) / true_val)

plt.scatter(np.arange(len(unique_counts)), actual_item_norms, c='r')
plt.scatter(np.arange(len(unique_counts)), pred_est_item_norms, c='b')
# plt.scatter(np.arange(len(unique_counts)), (pred_est_item_norms - actual_item_norms) / actual_item_norms, c='g')
# plt.scatter(np.arange(len(unique_counts)), np.abs(pred_est_item_norms - actual_item_norms) / actual_item_norms, c='b')

plt.yscale('log')
plt.xscale('log')


# +
##### Now do sketch where we have exact counts of heavy hitters

def simulate_loss_estimate_hh(est_freqs, n_estimate, k, k_hh, p, min_freq):
    buckets = generate_buckets(min_freq, k)

    # Filter out top k_hh in est_freqs
    top_khh_indices = np.argpartition(est_freqs, -k_hh)[-k_hh:]
    n_filtered = 0
    exact_norms = 0
    for i in top_khh_indices:
        n_filtered += unique_counts.iloc[i]
        exact_norms += unique_counts.iloc[i] ** p
        est_freqs.iloc[i] = -1
    
    k = len(buckets) - 1
    filled_buckets = [[] for _ in range(k)]
    for item, freq in est_freqs.items():
        if freq == -1:
            continue
        filled_buckets[find_bucket_index(buckets, freq)].append(unique_counts[item])
    for i in range(k):
        filled_buckets[i] = np.array(filled_buckets[i])
        
    bucket_norms = np.empty(k)
    bucket_preds = np.empty(k)

    for i, items in enumerate(filled_buckets):
        left, right = buckets[i] * len(df), buckets[i+1] * len(df)
        S = items.sum()

        n = n_estimate(S, left, right) if S > 0 else np.nan
        n = np.maximum(1, n)
        pred = (S**p)/(n**(p-1)) if S > 0 else np.nan
    
        bucket_norms[i] = norm(items, p)
        bucket_preds[i] = pred

    return exact_norms, bucket_norms, bucket_preds

est_freqs = est_rel_freqs(0.05)
exact_norms, bucket_norms, bucket_preds = simulate_loss_estimate_hh(est_freqs, n_estimate_harm_avg, 2**15, 2**15, 3, 7)
total_norms, total_preds = exact_norms + np.nan_to_num(bucket_norms).sum(), exact_norms + np.nan_to_num(bucket_preds).sum()
print((total_norms - total_preds) / total_norms)
print((true_value - total_preds) / total_preds)
