# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# %%
### Load data
def read_processed_data(file):
    df = pd.read_csv(f'data/processed/{file}', names=['id'])
    unique_counts = df['id'].value_counts()
    unique_freqs = unique_counts / len(df)
    return unique_counts, unique_freqs


aol_train_counts, aol_train_freqs = read_processed_data('AOL/train.txt')
aol_test_counts, aol_test_freqs = read_processed_data('AOL/test.txt')


# %%
# Generate Buckets
def expo_buckets(min_freq, k):
    step_size = min_freq / k
    buckets = np.arange(k) * -step_size
    buckets = 10 ** buckets
    buckets = np.append(buckets, 0)
    return buckets[::-1]


# %%
# Helper Functions
rng = np.random.default_rng()

def rel_oracle_est(freqs, ep, N):
    oracle_freqs = freqs.copy()
    err = rng.uniform(1-ep, 1+ep, len(freqs))
    i = 0
    for item, freq in oracle_freqs.items():
        oracle_freqs[item] = min(1, max(1/N, err[i] * freq))
        i += 1
    return oracle_freqs / np.sum(oracle_freqs)

def abs_oracle_est(freqs, ep, N):
    oracle_freqs = freqs.copy()
    err = rng.uniform(-ep, ep, len(freqs))
    i = 0
    for item, freq in oracle_freqs.items():
        oracle_freqs[item] = min(1, max(1/N, err[i] + freq))
        i += 1
    return oracle_freqs / np.sum(oracle_freqs)

def train_oracle_est(freqs, train_freqs, N):
    oracle_freqs = freqs.copy()
    for item, freq in oracle_freqs.items():
        if item in train_freqs:
            oracle_freqs[item] = train_freqs[item]
        else:
            oracle_freqs[item] = 1/N
    return oracle_freqs / np.sum(oracle_freqs)


# %%
def plot_per_key_error(buckets, est_freqs, counts, p, title, file_name):
    k = len(buckets) - 1
    N = np.sum(counts)
    item_est = np.empty(len(counts))
    item_counts = np.empty(len(counts))
    curr_item = 0
    
    sorted_freqs = est_freqs.sort_values(ascending=True)
    curr_bucket = 0
    for item, freq in sorted_freqs.items():
        while buckets[curr_bucket + 1] < freq:
            curr_bucket += 1
        left, right = buckets[curr_bucket] * N, buckets[curr_bucket + 1] * N
        item_est[curr_item] = counts[item] * ((left + right) / 2) ** (p-1)
        item_counts[curr_item] = counts[item]
        curr_item += 1

    item_real = item_counts ** p
    item_err_abs = np.abs(item_est - item_real)
    item_err_rel = (item_est - item_real) / item_real

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(15, 5))
    sns.scatterplot(ax=ax[0], x=item_counts / N, y=item_err_rel, alpha=0.5)
    sns.scatterplot(ax=ax[1], x=item_counts / N, y=item_err_abs, alpha=0.5)

    ax[0].set_xscale('log')
    ax[1].set_yscale('log')

    fig.supxlabel('Element Frequency (log-scale)', y=-0.05)
    ax[0].set_ylabel('Relative Error')
    ax[1].set_ylabel('Absolute Error (log-scale)')

    fig.suptitle(title)

    plt.savefig(f'figs/{file_name}', bbox_inches='tight')
    plt.show()

p = 3
k = 1024
N = np.sum(aol_test_counts)
buckets = expo_buckets(7, k)
plot_per_key_error(buckets, rel_oracle_est(aol_test_freqs, 0.05, N), aol_test_counts, p,
                  "Per-Key Relative and Absolute Error for 3rd Frequency Moment on AOL Data, under the Relative Error Oracle",
                  "3rd_moment_bucket_per_key_err_rel_aol")
plot_per_key_error(buckets, abs_oracle_est(aol_test_freqs, 0.001, N), aol_test_counts, p,
                  "Per-Key Relative and Absolute Error for 3rd Frequency Moment on AOL Data, under the Absolute Error Oracle",
                  "3rd_moment_bucket_per_key_err_abs_aol")
plot_per_key_error(buckets, train_oracle_est(aol_test_freqs, aol_train_freqs, N), aol_test_counts, p,
                  "Per-Key Relative and Absolute Error for 3rd Frequency Moment on AOL Data, under the Train/Test Oracle",
                  "3rd_moment_bucket_per_key_err_train_aol")
