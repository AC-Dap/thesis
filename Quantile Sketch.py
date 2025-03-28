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

fake_01_train_counts, fake_01_train_freqs = read_processed_data('fake_0.1_dataset/train.txt')
fake_01_test_counts, fake_01_test_freqs = read_processed_data('fake_0.1_dataset/test_1.txt')

fake_03_train_counts, fake_03_train_freqs = read_processed_data('fake_0.3_dataset/train.txt')
fake_03_test_counts, fake_03_test_freqs = read_processed_data('fake_0.3_dataset/test_1.txt')

fake_05_train_counts, fake_05_train_freqs = read_processed_data('fake_0.5_dataset/train.txt')
fake_05_test_counts, fake_05_test_freqs = read_processed_data('fake_0.5_dataset/test_1.txt')


# %%
def expo_buckets(min_freq, k):
    step_size = np.exp(-np.log(min_freq) / (k-2))
    buckets = 1 / step_size ** np.arange(k)
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
        new_freq = min(1, max(0, err[i] * freq))
        if new_freq == 0:
            new_freq = 1e-9
        oracle_freqs[item] = new_freq
        i += 1
    return oracle_freqs

def abs_oracle_est(freqs, ep, N):
    oracle_freqs = freqs.copy()
    err = rng.uniform(-ep, ep, len(freqs))
    i = 0
    for item, freq in oracle_freqs.items():
        new_freq = min(1, max(0, err[i] + freq))
        if new_freq == 0:
            new_freq = 1e-9
        oracle_freqs[item] = new_freq
        i += 1
    return oracle_freqs

def train_oracle_est(freqs, train_freqs, N):
    oracle_freqs = freqs.copy()
    for item, freq in oracle_freqs.items():
        if item in train_freqs:
            oracle_freqs[item] = train_freqs[item]
        else:
            oracle_freqs[item] = 1/N
    return oracle_freqs


# %%
def bucket_quantile_est(buckets, est_freqs, true_counts):
    k = len(buckets) - 1
    N = np.sum(true_counts)

    bucket_sum = np.zeros(k, dtype=float)
    bucket_min_count = np.full(k, np.inf, dtype=float)
    bucket_max_count = np.full(k, 0, dtype=float)

    curr_bucket = 0
    curr_sample = []
    sorted_freqs = est_freqs.sort_values(ascending=True)
    for item, f in sorted_freqs.items():
        while buckets[curr_bucket + 1] < f:
            if len(curr_sample) > 0:
                s = np.random.choice(curr_sample, min(len(curr_sample), 16))
                bucket_min_count[curr_bucket] = np.min(s)
                bucket_max_count[curr_bucket] = np.max(s)
            curr_sample = []
            curr_bucket += 1
        curr_sample.append(true_counts[item])
        bucket_sum[curr_bucket] += true_counts[item]
    
    s = np.random.choice(curr_sample, min(len(curr_sample), 16))
    bucket_min_count[curr_bucket] = np.min(s)
    bucket_max_count[curr_bucket] = np.max(s)

    return bucket_sum, bucket_min_count, bucket_max_count


# %%
def compute_f(s, N, M, m):
    k_M = int(np.floor((N / M)**(1/s)))
    k_m = int(np.ceil((N / m)**(1/s)))
    ks = np.arange(k_M, k_m + 1)
    total = np.sum(np.floor(N / (ks ** s)))
    return total

def find_s(target_f, N, M, m, tol=1e-5, s_min=0.5, s_max=5.0):
    while s_max - s_min > tol:
        s_mid = (s_min + s_max) / 2
        f_mid = compute_f(s_mid, N, M, m)
        
        if f_mid < target_f:
            s_max = s_mid
        else:
            s_min = s_mid
    
    return (s_min + s_max) / 2


# %%
def estimate_rank(x, buckets, bucket_sum):
    k = len(buckets) - 1
    N = np.sum(bucket_sum)
    count_est = 0
    num_smaller = 0
    for i in range(k):
        if bucket_sum[i] == 0:
            continue
        
        m, M = buckets[i] * N, buckets[i+1] * N
        n = bucket_sum[i] / ((m+M)/2)
        if x < m: 
            num_smaller += 0
        elif x >= M:
            num_smaller += n
        else:
            d = (M - m) / (n - 1)
            l = int(np.floor((x - m) / d))
            num_smaller += l + 1
        count_est += n
    return num_smaller / count_est

def est_rank_2(x, N, b_sums, b_mins, b_maxes, fitted_s):
    k = len(b_mins)
    num_smaller = 0
    count = 0
    for i in range(k):
        if b_sums[i] == 0:
            continue
        
        m, M, s = b_mins[i], b_maxes[i], fitted_s[i]
        if m == M:
            # Use exact count
            c = np.ceil(b_sums[i] / m)
            if x >= m:
                num_smaller += c
            count += c
            continue
        
        k_M = int(np.floor((N / M)**(1/s)))
        k_m = int(np.ceil((N / m)**(1/s)))
        
        # How many np.floor(N / (k**s)) <= x:
        threshold = (N / (x + 1)) ** (1 / s)
        l_min = np.floor(threshold) + 1
        l_min = max(l_min, k_M)
        
        if l_min <= k_m:
            num_smaller += k_m - l_min + 1

        count += k_m - k_M + 1
    return num_smaller / count


# %%
def plot_quantiles_bad(train_freqs, test_freqs, test_counts, title, file_name):
    N = np.sum(test_counts)
    rel_est_freqs = rel_oracle_est(test_freqs, 0.05, N)
    abs_est_freqs = abs_oracle_est(test_freqs, 0.001, N)
    train_est_freqs = train_oracle_est(test_freqs, train_freqs, N)

    print("Generated estimated freqs")

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    est_freqs = [rel_est_freqs, abs_est_freqs, train_est_freqs]
    error_types = ["Relative", "Absolute", "Train/test"]
    
    ks = [1024, 4096, 16384, 65536]
    for k in ks:
        for i in range(3):
            buckets = expo_buckets(np.min(est_freqs[i]), k)
            b_sums, b_mins, b_maxs = bucket_quantile_est(buckets, est_freqs[i], test_counts)
            
            xs = np.logspace(np.log10(np.min(test_counts)), np.log10(np.max(test_counts)), 100)
            qs_act = np.zeros(len(xs))
            qs_est = np.zeros(len(xs))
            for j, x in enumerate(xs):
                qs_act[j] = np.searchsorted(test_counts[::-1], x, side='right') / len(test_counts)
                qs_est[j] = estimate_rank(x, buckets, b_sums)
            
            err = np.abs(qs_act - qs_est)
            print(f"{k} {error_types[i]}: ", np.max(err))
            ax[i].plot(xs, err, label=f"k={k}", linestyle="dashed", marker='^', markersize=5)
            ax[i].set_title(f"{error_types[i]} error")

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    tick_marks = [f"{int(x)}" for x in xs[::10]]
    ax[0].set_xticks(xs[::10], tick_marks, rotation='vertical')
    ax[1].set_xticks(xs[::10], tick_marks, rotation='vertical')
    ax[2].set_xticks(xs[::10], tick_marks, rotation='vertical')

    fig.supxlabel('$w$', y=-0.1)
    fig.supylabel('Rank Error')

    fig.suptitle(title)

    fig.legend(*ax[0].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/{file_name}', bbox_inches='tight')
    plt.show()


# %%
plot_quantiles_bad(aol_train_freqs, aol_test_freqs, aol_test_counts, "Quantile Sketch Errors on AOL Data, using Naive estimator", "quantile_aol_bad")


# %%
def plot_quantiles(train_freqs, test_freqs, test_counts, title, file_name):
    N = np.sum(test_counts)
    rel_est_freqs = rel_oracle_est(test_freqs, 0.05, N)
    abs_est_freqs = abs_oracle_est(test_freqs, 0.001, N)
    train_est_freqs = train_oracle_est(test_freqs, train_freqs, N)

    print("Generated estimated freqs")

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    est_freqs = [rel_est_freqs, abs_est_freqs, train_est_freqs]
    error_types = ["Relative", "Absolute", "Train/test"]
    
    ks = [1024, 4096, 16384, 65536]
    for k in ks:
        for i in range(3):
            buckets = expo_buckets(np.min(est_freqs[i]), k)
            b_sums, b_mins, b_maxs = bucket_quantile_est(buckets, est_freqs[i], test_counts)
            
            fitted_s = np.zeros(k)
            for j in range(k):
                if b_sums[j] > 0 and b_mins[j] != b_maxs[j]:
                    fitted_s[j] = find_s(b_sums[j], N, b_maxs[j], b_mins[j])
    
            xs = np.logspace(np.log10(np.min(test_counts)), np.log10(np.max(test_counts)), 100)
            qs_act = np.zeros(len(xs))
            qs_est = np.zeros(len(xs))
            for j, x in enumerate(xs):
                qs_act[j] = np.searchsorted(test_counts[::-1], x, side='right') / len(test_counts)
                qs_est[j] = est_rank_2(x, N, b_sums, b_mins, b_maxs, fitted_s)
            
            err = np.abs(qs_act - qs_est)
            print(f"{k} {error_types[i]}: ", np.max(err))
            ax[i].plot(xs, err, label=f"k={k}", linestyle="dashed", marker='^', markersize=5)
            ax[i].set_title(f"{error_types[i]} error")

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    tick_marks = [f"{int(x)}" for x in xs[::10]]
    ax[0].set_xticks(xs[::10], tick_marks, rotation='vertical')
    ax[1].set_xticks(xs[::10], tick_marks, rotation='vertical')
    ax[2].set_xticks(xs[::10], tick_marks, rotation='vertical')

    fig.supxlabel('$w$', y=-0.1)
    fig.supylabel('Rank Error')

    fig.suptitle(title)

    fig.legend(*ax[0].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/{file_name}', bbox_inches='tight')
    plt.show()


# %%
plot_quantiles(aol_train_freqs, aol_test_freqs, aol_test_counts, "Quantile Sketch Errors on AOL Data", "quantile_aol")
plot_quantiles(fake_01_train_freqs, fake_01_test_freqs, fake_01_test_counts, "Quantile Sketch Errors on Synthetic $\\alpha=0.1$ Data", "quantile_fake_0_1")
plot_quantiles(fake_03_train_freqs, fake_03_test_freqs, fake_03_test_counts, "Quantile Sketch Errors on Synthetic $\\alpha=0.3$ Data", "quantile_fake_0_3")
plot_quantiles(fake_05_train_freqs, fake_05_test_freqs, fake_05_test_counts, "Quantile Sketch Errors on Synthetic $\\alpha=0.5$ Data", "quantile_fake_0_5")
