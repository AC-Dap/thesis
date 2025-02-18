# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
def load_data(path):
    df = pd.read_csv(path, sep='\t')
    df['Query'] = df['Query'].fillna("")
    unique_counts = df['Query'].value_counts()
    return df, unique_counts

df_train, unique_counts_train = load_data('data/fake_0.1_dataset/train.txt')
# df_train, unique_counts_train = load_data('data/AOL-user-ct-collection/user-ct-test-collection-01.txt')
df, unique_counts = load_data('data/fake_0.1_dataset/test_1.txt')
# df, unique_counts = load_data('data/AOL-user-ct-collection/user-ct-test-collection-02.txt')

aol_df, aol_unique_counts = load_data('data/AOL-user-ct-collection/user-ct-test-collection-01.txt')
fake_0_1_df, fake_0_1_unique_counts = load_data('data/fake_0.1_dataset/train.txt')
fake_0_3_df, fake_0_3_unique_counts = load_data('data/fake_0.3_dataset/train.txt')
fake_0_5_df, fake_0_5_unique_counts = load_data('data/fake_0.5_dataset/train.txt')

# %%
plt.plot(np.arange(len(aol_unique_counts)), aol_unique_counts / len(aol_df), label="AOL")
plt.plot(np.arange(len(fake_0_1_unique_counts)), fake_0_1_unique_counts / len(fake_0_1_df), label="Synthetic $\\alpha=0.1$")
plt.plot(np.arange(len(fake_0_3_unique_counts)), fake_0_3_unique_counts / len(fake_0_3_df), label="Synthetic $\\alpha=0.3$")
plt.plot(np.arange(len(fake_0_5_unique_counts)), fake_0_5_unique_counts / len(fake_0_5_df), label="Synthetic $\\alpha=0.5$")

plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Rank vs. Frequency of datasets')
plt.savefig('figs/data_rank_freq', bbox_inches='tight')

# %%
pareto_dataset = np.floor(100 * (1 + np.random.pareto(0.5, size=1000000)))
print(len(pareto_dataset))
print(len(df_train))
_, unique_counts = np.unique(pareto_dataset, return_counts=True)
print(len(unique_counts))
plt.plot(np.arange(len(unique_counts)), unique_counts / len(pareto_dataset))
plt.plot(np.arange(len(unique_counts_train)), unique_counts_train / len(df_train))

plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Rank vs. Frequency of search results')
plt.savefig('figs/aol_rank_frequency')

# %%
oracle_guesses = np.empty(len(unique_counts))
i = 0
for value, count in unique_counts.items():
    oracle_guesses[i] = unique_counts_train[value] if (value in unique_counts_train) else 1
    i += 1
plt.plot(unique_counts / len(df), oracle_guesses / len(df_train))
plt.plot(unique_counts / len(df), oracle_guesses / len(df))
# plt.plot(unique_counts / len(df), unique_counts / len(df), c='r')

# plt.plot(unique_counts / len(df), np.abs((unique_counts / len(df)) - (oracle_guesses / len(df_train))), c='r')

# %%
sim_folder = "simulation/results"

deg3_aol, deg4_aol = pd.read_csv(f"{sim_folder}/new_deg=3_aol_results.csv"), pd.read_csv(f"{sim_folder}/new_deg=4_aol_results.csv")
deg3_0_1, deg4_0_1 = pd.read_csv(f"{sim_folder}/new_deg=3_fake_0.1_results.csv"), pd.read_csv(f"{sim_folder}/new_deg=4_fake_0.1_results.csv")
deg3_0_3, deg4_0_3 = pd.read_csv(f"{sim_folder}/new_deg=3_fake_0.3_results.csv"), pd.read_csv(f"{sim_folder}/new_deg=4_fake_0.3_results.csv")
deg3_0_5, deg4_0_5 = pd.read_csv(f"{sim_folder}/new_deg=3_fake_0.5_results.csv"), pd.read_csv(f"{sim_folder}/new_deg=4_fake_0.5_results.csv")

threshold_aol = pd.read_csv(f"{sim_folder}/threshold_aol_results.csv")
threshold_fake_0_1 = pd.read_csv(f"{sim_folder}/threshold_fake_0.1_results.csv")
threshold_fake_0_3 = pd.read_csv(f"{sim_folder}/threshold_fake_0.3_results.csv")
threshold_fake_0_5 = pd.read_csv(f"{sim_folder}/threshold_fake_0.5_results.csv")

nsims = 30
sample_sizes = [2**i for i in range(6, 17)]
def read_sim_data(df, sketch_type, sample_sizes):
    estimates = []
    exacts = []
    for k in sample_sizes:
        mask = (df["sketch_type"] == sketch_type) & (df["k"] == k)
        estimates.append(df.loc[mask, "estimate"].apply(float.fromhex).to_numpy())
        exacts.append(df.loc[mask, "exact"].apply(float).to_numpy())
    return np.array(estimates), np.array(exacts)
    
def nth(deg):
    return "3rd" if deg == 3 else f"{deg}th"

def error_prefix(err):
    if err == "rel_0.05": return "relative"
    if err == "abs_0.001": return "absolute"
    if err == "train": return "train/test"
    if err == "exact": return "no"


# %%
def plot_curve(ax, x, results, true_value, label, color):
    error = (results - true_value) / true_value
    mean = np.mean(error, axis=1)
    lower = np.min(error, axis=1)
    upper = np.max(error, axis=1)
    
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)
    ax.legend()

def plot_abs_curve(ax, x, results, true_value, label, color):
    error = (results - true_value) / true_value
    mean = np.mean(np.abs(error), axis=1)
    upper = np.max(np.abs(error), axis=1)
    
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean, upper, color=color, alpha=0.2)
    ax.legend()

def plot_mse(ax, x, results, true_value, label):
    mse = np.sqrt(np.mean(((results - true_value) / true_value)**2, axis=1))
    
    ax.plot(x, mse, label=label)

# %%
# Plot NRMSE error for SWA, per oracle error type

# %%
# Plot NRMSE error for SWA, per oracle error type

def plot_swa_nrmse(df, df_name, df_prefix):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    error_types = ["rel_0.05", "abs_0.001", "train"]

    for i, error_type in enumerate(error_types):
        swa_1_results, swa_1_exacts = read_sim_data(df, f"swa_{error_type}_kh=0_kp=k_ku=0", sample_sizes)
        swa_2_results, swa_2_exacts = read_sim_data(df, f"swa_{error_type}_kh=k/2_kp=k/2_ku=0", sample_sizes)
        swa_3_results, swa_3_exacts = read_sim_data(df, f"swa_{error_type}_kh=0_kp=k/2_ku=k/2", sample_sizes)

        plot_mse(ax[i], sample_sizes, swa_1_results, swa_1_exacts, "$k_h = 0, k_p = k, k_u = 0$")
        plot_mse(ax[i], sample_sizes, swa_2_results, swa_2_exacts, "$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$")
        plot_mse(ax[i], sample_sizes, swa_3_results, swa_3_exacts, "$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$")

        error_name = error_prefix(error_type)
        ax[i].set_title(f'{error_name} error')

    # ax[0].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    # ax[1].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Sample size', y=-0.1)
    fig.supylabel('Normalized Root MSE')

    fig.suptitle(f'Threshold Count NRMSE vs. Sample Size on {df_name} data, SWA Sketch')

    fig.legend(*ax[2].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/threshold_swa_{df_prefix}_nrmse', bbox_inches='tight')
    plt.show()

plot_swa_nrmse(threshold_aol, 'AOL', 'aol')
plot_swa_nrmse(threshold_fake_0_1, 'Synthetic $\\alpha=0.1$', 'fake_0_1')
plot_swa_nrmse(threshold_fake_0_3, 'Synthetic $\\alpha=0.3$', 'fake_0_3')
plot_swa_nrmse(threshold_fake_0_5, 'Synthetic $\\alpha=0.5$', 'fake_0_5')


# %%
# Plot estimation error for buckets, per oracle error type

def plot_bucket_nrmse(df, df_name, df_prefix):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 5))
    error_types = ["rel_0.05", "abs_0.001", "train"]
    bucket_types = ["linear", "expo"]

    for i, bucket_type in enumerate(bucket_types):
        for j, error_type in enumerate(error_types):
            # bucket_expo_harm1_results, bucket_expo_harm1_exacts = read_sim_data(df, f"bucket_{bucket_type}_harm_{error_type}_k=k_kh=0", sample_sizes)
            # bucket_expo_harm2_results, bucket_expo_harm2_exacts = read_sim_data(df, f"bucket_{bucket_type}_harm_{error_type}_k=k/2_kh=k/2", sample_sizes)

            bucket_expo_alt1_results, bucket_expo_alt1_exacts = read_sim_data(df, f"bucket_{bucket_type}_alt_{error_type}_k=k_kh=0", sample_sizes)
            bucket_expo_alt2_results, bucket_expo_alt2_exacts = read_sim_data(df, f"bucket_{bucket_type}_alt_{error_type}_k=k/2_kh=k/2", sample_sizes)

            bucket_expo_smart1_results, bucket_expo_smart1_exacts = read_sim_data(df, f"smart_a_expo_arith_{error_type}_k=k/2_kh=k/2", sample_sizes)
            bucket_expo_smart2_results, bucket_expo_smart2_exacts = read_sim_data(df, f"smart_b_expo_arith_{error_type}_k=k/2_kh=k/2", sample_sizes)

            # bucket_expo_swa_results, bucket_expo_swa_exacts = read_sim_data(df, f"swa_bucket_{bucket_type}_{error_type}_k=sqrt(k)_kp=sqrt(k)_kh=0", sample_sizes)
            # bucket_expo_unif_results, bucket_expo_unif_exacts = read_sim_data(df, f"unif_bucket_{bucket_type}_{error_type}_k=sqrt(k)_ku=sqrt(k)_kh=0", sample_sizes)

            # bucket_cond_results, bucket_cond_exacts = read_sim_data(df, f"cond_bucket_{error_type}_k=1_kh=k", sample_sizes)

            # plot_mse(ax[i][j], sample_sizes, bucket_expo_harm1_results, bucket_expo_harm1_exacts, "Avg: $B = k, k_h = 0$", 'b')
            # plot_mse(ax[i][j], sample_sizes, bucket_expo_harm2_results, bucket_expo_harm2_exacts, "Avg: $B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'r')

            plot_mse(ax[i][j], sample_sizes, bucket_expo_alt1_results, bucket_expo_alt1_exacts, "Alt: $B = k, k_h = 0$")
            plot_mse(ax[i][j], sample_sizes, bucket_expo_alt2_results, bucket_expo_alt2_exacts, "Alt: $B = \\frac{k}{2}, k_h = \\frac{k}{2}$")

            plot_mse(ax[i][j], sample_sizes, bucket_expo_smart1_results, bucket_expo_smart1_exacts, "Smart A: $B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
            plot_mse(ax[i][j], sample_sizes, bucket_expo_smart2_results, bucket_expo_smart2_exacts, "Smart B: $B = \\frac{k}{2}, k_h = \\frac{k}{2}$")

            # plot_mse(ax[i][j], sample_sizes, bucket_expo_swa_results, bucket_expo_swa_exacts, "SWA: $B = \\sqrt{k}, k_h = \\sqrt{k}$", 'pink')
            # plot_mse(ax[i][j], sample_sizes, bucket_expo_unif_results, bucket_expo_unif_exacts, "Unif: $B = \\sqrt{k}, k_h = \\sqrt{k}$", 'orange')

            # plot_mse(ax[i][j], sample_sizes, bucket_cond_results, bucket_cond_exacts, "Unbiased: $k_h = k$", 'pink')

            error_name = error_prefix(error_type)
            bucket_name = "Linear" if bucket_type == "linear" else "Exponential"
            ax[i][j].set_title(f'{bucket_name} bucket, {error_name} error')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')

    ax[1][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Sample size', y=-0.1)
    fig.supylabel('Normalized Root MSE')

    fig.suptitle(f'Threshold Count NRMSE vs. Sample Size on {df_name} data, Bucketing Sketches')

    fig.legend(*ax[0][0].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/threshold_bucket_{df_prefix}_nrmse', bbox_inches='tight')
    plt.show()

plot_bucket_nrmse(threshold_aol, 'AOL', 'aol')
# plot_bucket_nrmse(threshold_fake_0_1, 'Synthetic $\\alpha=0.1$', 'fake_0_1')
# plot_bucket_nrmse(threshold_fake_0_3, 'Synthetic $\\alpha=0.3$', 'fake_0_3')
# plot_bucket_nrmse(threshold_fake_0_5, 'Synthetic $\\alpha=0.5$', 'fake_0_5')

# %%
# Plot NRMSE error comparing both sketches, per oracle error type

def plot_both_nrmse(df, df_name, df_prefix):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    error_types = ["rel_0.05", "abs_0.001", "train"]

    for i, error_type in enumerate(error_types):
        swa1_results, swa1_exacts = read_sim_data(df, f"swa_{error_type}_kh=k/2_kp=k/2_ku=0", sample_sizes)
        swa2_results, swa2_exacts = read_sim_data(df, f"swa_{error_type}_kh=0_kp=k/2_ku=k/2", sample_sizes)
        # bucket_expo_harm_results, bucket_expo_harm_exacts = read_sim_data(df, f"bucket_expo_harm_{error_type}_k=k/2_kh=k/2", sample_sizes)
        bucket_expo_alt1_results, bucket_expo_alt1_exacts = read_sim_data(df, f"bucket_expo_alt_{error_type}_k=k_kh=0", sample_sizes)
        bucket_expo_alt2_results, bucket_expo_alt2_exacts = read_sim_data(df, f"bucket_expo_alt_{error_type}_k=k/2_kh=k/2", sample_sizes)
        # bucket_cond_results, bucket_cond_exacts = read_sim_data(df, f"cond_bucket_{error_type}_k=1_kh=k", sample_sizes)

        plot_mse(ax[i], sample_sizes, swa1_results, swa1_exacts, "SWA: $k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$")
        plot_mse(ax[i], sample_sizes, swa2_results, swa2_exacts, "SWA: $k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$")
        # plot_mse(ax[i], sample_sizes, bucket_expo_harm_results, bucket_expo_harm_exacts, "Exponential Bucket Avg:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'g')
        plot_mse(ax[i], sample_sizes, bucket_expo_alt1_results, bucket_expo_alt1_exacts, "Exponential Bucket Alt:\n$B = k, k_h = 0$")
        plot_mse(ax[i], sample_sizes, bucket_expo_alt2_results, bucket_expo_alt2_exacts, "Exponential Bucket Alt:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
        # plot_mse(ax[i], sample_sizes, bucket_cond_results, bucket_cond_exacts, "Bucket Unbiased: $k_h = k$", 'pink')

        error_name = error_prefix(error_type)
        ax[i].set_title(f'{error_name} error')

    # ax[0].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    # ax[1].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Sample size', y=-0.1)
    fig.supylabel('Normalized Root MSE')

    fig.suptitle(f'Threshold Count NRMSE vs. Sample Size on {df_name} data, Comparing Sketches')

    fig.legend(*ax[2].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/threshold_both_{df_prefix}_nrmse', bbox_inches='tight')
    plt.show()

plot_both_nrmse(threshold_aol, 'AOL', 'aol')
plot_both_nrmse(threshold_fake_0_1, 'Synthetic $\\alpha=0.1$', 'fake_0_1')
plot_both_nrmse(threshold_fake_0_3, 'Synthetic $\\alpha=0.3$', 'fake_0_3')
plot_both_nrmse(threshold_fake_0_5, 'Synthetic $\\alpha=0.5$', 'fake_0_5')


# %%
# Plot absolute estimation error vs. sample size

def plot_error_sample_size(deg, error_type):
    ep = 0.05

    prefix = error_prefix(error_type)
    
    ppswor_results, ppswor_exacts = read_sim_data(deg, "ppswor", sample_sizes)
    swa_1_results, swa_1_exacts = read_sim_data(deg, f"swa_{prefix}_kh=0_kp=k_ku=0", sample_sizes)
    swa_2_results, swa_2_exacts = read_sim_data(deg, f"swa_{prefix}_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_3_results, swa_3_exacts = read_sim_data(deg, f"swa_{prefix}_kh=0_kp=k/2_ku=k/2", sample_sizes)
    
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 5))
    
    plot_curve(ax[0], sample_sizes, ppswor_results, ppswor_exacts, "PPSWOR", 'b')
    plot_curve(ax[1], sample_sizes, swa_1_results, swa_1_exacts, "SWA: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_curve(ax[2], sample_sizes, swa_2_results, swa_2_exacts, "SWA: $k_h = 0$,\n$k_u = k_p$", 'r')
    plot_curve(ax[3], sample_sizes, swa_3_results, swa_3_exacts, "SWA: $k_h = k_p$,\n$k_u = 0$", 'g')
    
    # ax[0].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    # ax[1].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    
    ax[0].set_xscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[3].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    fig.supxlabel('Sample size')
    fig.supylabel(f'Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Estimation Error vs. Sample Size, using {ep} {error_type} error')
    
    # plt.savefig(f'figs/freq_sketch_sim_{deg}_moment_{prefix}_err_large_sample')
    plt.show()

# plot_error_sample_size(3, "relative")
# plot_error_sample_size(3, "absolute")
# plot_error_sample_size(3, "binomial")
# plot_error_sample_size(4, "relative")
# plot_error_sample_size(4, "absolute")
# plot_error_sample_size(4, "binomial")


# %%
# Plot absolute estimation error vs. sample size

def plot_error_sample_size(deg, error_type):
    ep = 0.05

    prefix = error_prefix(error_type)
    
    ppswor_results, ppswor_exacts = read_sim_data(deg, "ppswor", sample_sizes)
    swa_1_results, swa_1_exacts = read_sim_data(deg, f"swa_{prefix}_kh=0_kp=k_ku=0", sample_sizes)
    swa_2_results, swa_2_exacts = read_sim_data(deg, f"swa_{prefix}_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_3_results, swa_3_exacts = read_sim_data(deg, f"swa_{prefix}_kh=0_kp=k/2_ku=k/2", sample_sizes)
    
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0], sample_sizes, ppswor_results, ppswor_exacts, "PPSWOR", 'b')
    plot_abs_curve(ax[1], sample_sizes, swa_1_results, swa_1_exacts, "SWA: $k_h = 0$,\n$k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[2], sample_sizes, swa_2_results, swa_2_exacts, "SWA: $k_h = 0$,\n$k_p = \\frac{k}{2}, k_u= \\frac{k}{2}$", 'r')
    plot_abs_curve(ax[3], sample_sizes, swa_3_results, swa_3_exacts, "SWA: $k_h = \\frac{k}{2}$,\n$k_p = \\frac{k}{2}, k_u = 0$", 'g')
    
    # ax[0].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    # ax[1].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[3].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    fig.supxlabel('Sample size ($k$)')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Absolute Estimation Error vs. Sample Size, using {ep} {error_type} error')

    plt.tight_layout()
    # plt.savefig(f'figs/freq_sketch_sim_{nth(deg)}_moment_absolute_{prefix}_err')
    plt.show()

plot_error_sample_size(3, "relative")
plot_error_sample_size(3, "absolute")
plot_error_sample_size(3, "binomial")
plot_error_sample_size(4, "relative")
plot_error_sample_size(4, "absolute")
plot_error_sample_size(4, "binomial")


# %%
# Plot estimation error across oracle types

def plot_error_oracle_types(deg):
    ep = 0.05

    swa_rel_1_results, swa_rel_1_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k_ku=0", sample_sizes)
    swa_rel_2_results, swa_rel_2_exacts = read_sim_data(deg, "swa_rel_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_rel_3_results, swa_rel_3_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k/2_ku=k/2", sample_sizes)

    swa_abs_1_results, swa_abs_1_exacts = read_sim_data(deg, "swa_abs_kh=0_kp=k_ku=0", sample_sizes)
    swa_abs_2_results, swa_abs_2_exacts = read_sim_data(deg, "swa_abs_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_abs_3_results, swa_abs_3_exacts = read_sim_data(deg, "swa_abs_kh=0_kp=k/2_ku=k/2", sample_sizes)

    swa_bin_1_results, swa_bin_1_exacts = read_sim_data(deg, "swa_bin_kh=0_kp=k_ku=0", sample_sizes)
    swa_bin_2_results, swa_bin_2_exacts = read_sim_data(deg, "swa_bin_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_bin_3_results, swa_bin_3_exacts = read_sim_data(deg, "swa_bin_kh=0_kp=k/2_ku=k/2", sample_sizes)
    
    fig, ax = plt.subplots(3, 3,  sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0][0], sample_sizes, swa_rel_1_results, swa_rel_1_exacts, "Relative Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[0][1], sample_sizes, swa_rel_2_results, swa_rel_2_exacts, "Relative Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'b')
    plot_abs_curve(ax[0][2], sample_sizes, swa_rel_3_results, swa_rel_3_exacts, "Relative Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')

    plot_abs_curve(ax[1][0], sample_sizes, swa_abs_1_results, swa_abs_1_exacts, "Absolute Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[1][1], sample_sizes, swa_abs_2_results, swa_abs_2_exacts, "Absolute Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'b')
    plot_abs_curve(ax[1][2], sample_sizes, swa_abs_3_results, swa_abs_3_exacts, "Absolute Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')

    plot_abs_curve(ax[2][0], sample_sizes, swa_bin_1_results, swa_bin_1_exacts, "Binomial Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[2][1], sample_sizes, swa_bin_2_results, swa_bin_2_exacts, "Binomial Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'b')
    plot_abs_curve(ax[2][2], sample_sizes, swa_bin_3_results, swa_bin_3_exacts, "Binomial Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')

    ax[2][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2][2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    fig.supxlabel('Sample size ($k$)')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Estimation Error vs. Error Type')
    
    # plt.savefig(f'figs/freq_sketch_sim_{deg}_moment_error_type')
    plt.show()

# plot_error_oracle_types(3)
# plot_error_oracle_types(4)


# %%
# Plot estimation error across oracle types

def plot_error_oracle_types(deg):
    ep = 0.05
    
    exact_results, exact_exacts = read_sim_data(deg, "exact", sample_sizes)
    swa_rel_results, swa_rel_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k_ku=0", sample_sizes)
    swa_abs_results, swa_abs_exacts = read_sim_data(deg, "swa_abs_kh=0_kp=k_ku=0", sample_sizes)
    swa_bin_results, swa_bin_exacts = read_sim_data(deg, "swa_bin_kh=0_kp=k_ku=0", sample_sizes)
    
    fig, ax = plt.subplots(2, 2,  sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0][0], sample_sizes, exact_results, exact_exacts, "Exact", 'b')
    plot_abs_curve(ax[0][1], sample_sizes, swa_rel_results, swa_rel_exacts, "Relative: $k_h = 0$,\n$k_p = k, k_u = 0$", 'r')
    plot_abs_curve(ax[1][0], sample_sizes, swa_abs_results, swa_abs_exacts, "Absolute: $k_h = 0$,\n$k_p = k, k_u = 0$", 'g')
    plot_abs_curve(ax[1][1], sample_sizes, swa_bin_results, swa_bin_exacts, "Binomial: $k_h = 0$,\n$k_p = k, k_u = 0$", 'y')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')

    ax[1][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    fig.supxlabel('Sample size ($k$)')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Absolute Estimation Error vs. Error Type')

    plt.tight_layout()
    # plt.savefig(f'figs/freq_sketch_sim_{nth(deg)}_moment_error_type')
    plt.show()

plot_error_oracle_types(3)
plot_error_oracle_types(4)


# %%
# Plot MSE of simulations across epsilons

def plot_mse_rel(deg):
    ppswor_results, ppswor_exacts = read_sim_data(deg, "ppswor", sample_sizes)
    exact_results, exact_exacts = read_sim_data(deg, "exact", sample_sizes)
    
    swa_rel_1_results, swa_rel_1_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k_ku=0", sample_sizes)
    swa_rel_2_results, swa_rel_2_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k/2_ku=k/2", sample_sizes)
    swa_rel_3_results, swa_rel_3_exacts = read_sim_data(deg, "swa_rel_kh=k/2_kp=k/2_ku=0", sample_sizes)

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, ppswor_results, ppswor_exacts, "PPSWOR", 'r')
    plot_mse(ax, sample_sizes, exact_results, exact_exacts, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_1_results, swa_rel_1_exacts, "SWA ep=0.05: $k_h = 100$,\n$k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_2_results, swa_rel_2_exacts, "SWA ep=0.05: $k_h = 100$,\n$k_u = k_p$", 'y')
    plot_mse(ax, sample_sizes, swa_rel_3_results, swa_rel_3_exacts, "SWA ep=0.2: $k_h = 100$,\n$k_u = 0$", 'orange')
    
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')

    plt.xlabel('Sample size')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Epsilon')

    # plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_epsilon', bbox_inches='tight')
    plt.show()

# plot_mse_rel(3)
# plot_mse_rel(4)


# %%
# Plot MSE of simulations across error types

def plot_mse_error(deg):
    ppswor_results, ppswor_exacts = read_sim_data(deg, "ppswor", sample_sizes)
    exact_results, exact_exacts = read_sim_data(deg, "exact", sample_sizes)

    swa_rel_1_results, swa_rel_1_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k_ku=0", sample_sizes)
    swa_rel_2_results, swa_rel_2_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k/2_ku=k/2", sample_sizes)
    swa_rel_3_results, swa_rel_3_exacts = read_sim_data(deg, "swa_rel_kh=k/2_kp=k/2_ku=0", sample_sizes)

    swa_abs_1_results, swa_abs_1_exacts = read_sim_data(deg, "swa_abs_kh=0_kp=k_ku=0", sample_sizes)
    swa_abs_2_results, swa_abs_2_exacts = read_sim_data(deg, "swa_abs_kh=0_kp=k/2_ku=k/2", sample_sizes)
    swa_abs_3_results, swa_abs_3_exacts = read_sim_data(deg, "swa_abs_kh=k/2_kp=k/2_ku=0", sample_sizes)

    swa_bin_1_results, swa_bin_1_exacts = read_sim_data(deg, "swa_bin_kh=0_kp=k_ku=0", sample_sizes)
    swa_bin_2_results, swa_bin_2_exacts = read_sim_data(deg, "swa_bin_kh=0_kp=k/2_ku=k/2", sample_sizes)
    swa_bin_3_results, swa_bin_3_exacts = read_sim_data(deg, "swa_bin_kh=k/2_kp=k/2_ku=0", sample_sizes)
    
    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, ppswor_results, ppswor_exacts, "PPSWOR", 'r')
    plot_mse(ax, sample_sizes, exact_results, exact_exacts, "Exact", 'g')

    plot_mse(ax, sample_sizes, swa_rel_1_results, swa_rel_1_exacts, "Relative Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_2_results, swa_rel_2_exacts, "Relative Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'black')
    plot_mse(ax, sample_sizes, swa_rel_3_results, swa_rel_3_exacts, "Relative Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'grey')

    # plot_mse(ax, sample_sizes, swa_abs_1_results, swa_abs_1_exacts, "Absolute Error: \n$k_h = 0, k_p = k, k_u = 0$", 'y')
    # plot_mse(ax, sample_sizes, swa_abs_2_results, swa_abs_2_exacts, "Absolute Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'purple')
    # plot_mse(ax, sample_sizes, swa_abs_3_results, swa_abs_3_exacts, "Absolute Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'pink')

    plot_mse(ax, sample_sizes, swa_bin_1_results, swa_bin_1_exacts, "Binomial Error: \n$k_h = 0, k_p = k, k_u = 0$", 'cyan')
    plot_mse(ax, sample_sizes, swa_bin_2_results, swa_bin_2_exacts, "Binomial Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'gold')
    plot_mse(ax, sample_sizes, swa_bin_3_results, swa_bin_3_exacts, "Binomial Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'orange')

    
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    plt.xlabel('Sample size')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Error Type')

    # plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_error_type', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)


# %%
# Plot MSE of simulations across error types

def plot_mse_error(deg):
    exact_results, exact_exacts = read_sim_data(deg, "exact", sample_sizes)
    
    swa_rel_1_results, swa_rel_1_exacts = read_sim_data(deg, "swa_rel_kh=0_kp=k_ku=0", sample_sizes)
    swa_rel_2_results, swa_rel_2_exacts = read_sim_data(deg, "swa_rel_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_bin_1_results, swa_bin_1_exacts = read_sim_data(deg, "swa_bin_kh=0_kp=k_ku=0", sample_sizes)
    swa_bin_2_results, swa_bin_2_exacts = read_sim_data(deg, "swa_bin_kh=k/2_kp=k/2_ku=0", sample_sizes)

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, exact_results, exact_exacts, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_1_results, swa_rel_1_exacts, "Relative Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_2_results, swa_rel_2_exacts, "Relative Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'r')
    plot_mse(ax, sample_sizes, swa_bin_1_results, swa_bin_1_exacts, "Binomial Error: \n$k_h = 0, k_p = k, k_u = 0$", 'grey')
    plot_mse(ax, sample_sizes, swa_bin_2_results, swa_bin_2_exacts, "Binomial Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'cyan')

    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    plt.xlabel('Sample size ($k$)')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Error Type')

    # plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_error_type_focused', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)


# %%
# Plot MSE of simulations across error types

def plot_mse_error(deg):
    exact_results, exact_exacts = read_sim_data(deg, "exact", sample_sizes)
    
    swa_rel_results, swa_rel_exacts = read_sim_data(deg, "swa_rel_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_bin_results, swa_bin_exacts = read_sim_data(deg, "swa_bin_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_abs_results, swa_abs_exacts = read_sim_data(deg, "swa_abs_kh=k/2_kp=k/2_ku=0", sample_sizes)

    n_est_type = "arith"
    expo_rel_1, expo_rel_1_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_rel_k=k/2_kh=k/2", sample_sizes)
    expo_bin_1, expo_bin_1_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_bin_k=k/2_kh=k/2", sample_sizes)
    expo_abs_1, expo_abs_1_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_abs_k=k/2_kh=k/2", sample_sizes)

    expo_rel_2, expo_rel_2_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_rel_k=k_kh=0", sample_sizes)
    expo_bin_2, expo_bin_2_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_bin_k=k_kh=0", sample_sizes)
    expo_abs_2, expo_abs_2_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_abs_k=k_kh=0", sample_sizes)
    
    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, exact_results, exact_exacts, "Exact", 'g')
    
    plot_mse(ax, sample_sizes, swa_rel_results, swa_rel_exacts, "SWA: Relative Error,\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'r')
    plot_mse(ax, sample_sizes, swa_bin_results, swa_bin_exacts, "SWA: Binomial Error,\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'cyan')
    plot_mse(ax, sample_sizes, swa_abs_results, swa_abs_exacts, "SWA: Absolute Error,\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')


    plot_mse(ax, sample_sizes, expo_rel_1, expo_rel_1_exacts, "Bucket: Relative Error,\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'gold')
    plot_mse(ax, sample_sizes, expo_bin_1, expo_bin_1_exacts, "Bucket: Binomial Error,\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'darkkhaki')
    plot_mse(ax, sample_sizes, expo_abs_1, expo_abs_1_exacts, "Bucket: Absolute Error,\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'purple')

    plot_mse(ax, sample_sizes, expo_rel_2, expo_rel_2_exacts, "Bucket: Relative Error,\n$B = k, k_h = 0$", 'pink')
    plot_mse(ax, sample_sizes, expo_bin_2, expo_bin_2_exacts, "Bucket: Binomial Error,\n$B = k, k_h = 0$", 'black')
    plot_mse(ax, sample_sizes, expo_abs_2, expo_abs_2_exacts, "Bucket: Absolute Error,\n$B = k, k_h = 0$", 'grey')

    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    plt.xlabel('Sample size ($k$)')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE, SWA vs. Bucketing Sketch')

    plt.savefig(f'figs/freq_bucket_sketch_sim_nrmse_{nth(deg)}_moment_error_type_focused', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)


# %%
def plot_bucketing_sketches(deg):
    fig, ax = plt.subplots(2, 3,  sharex=True, sharey=True, figsize=(12, 5))

    for i, bucket_type in enumerate(["linear", "expo"]):
        for j, oracle_type in enumerate(["rel", "abs", "bin"]):
            c = [('r', 'g'), ('b', 'y'), ('maroon', 'black'), ('gold', 'pink'), ('purple', 'cyan')]
            est_type_label = ["Lower bound", "Upper bound", "Arithmetic mean", "Geometric mean", "Harmonic mean"]
            for l, n_est_type in enumerate(["lower", "upper", "arith", "geo", "harm"]):
                no_hh, no_hh_exacts = read_sim_data(deg, f"bucket_{bucket_type}_{n_est_type}_{oracle_type}_k=k_kh=0", sample_sizes)
                with_hh, with_hh_exacts = read_sim_data(deg, f"bucket_{bucket_type}_{n_est_type}_{oracle_type}_k=k/2_kh=k/2", sample_sizes)

                plot_abs_curve(ax[i][j], sample_sizes, no_hh, no_hh_exacts, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
                plot_abs_curve(ax[i][j], sample_sizes, with_hh, with_hh_exacts, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])
            ax[i][j].set_title(f"{bucket_type} {oracle_type}")
            ax[i][j].get_legend().remove()
            # sns.move_legend(ax[i][j], "upper left", bbox_to_anchor=(1, 1))

    ax[0][0].set_yscale('log')
    ax[0][0].set_xscale('log')

    ax[1][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Space size ($k$)')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Absolute Estimation Error vs. Space size, Bucketing Sketch')
    
    plt.tight_layout()

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

    # plt.savefig(f'figs/bucket_sketch_sim_{nth(deg)}_moment_absolute_err', bbox_inches='tight')
    plt.show()

plot_bucketing_sketches(3)
plot_bucketing_sketches(4)


# %%
def plot_bucketing_kh_sketches(deg):
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 6))

    c = [('r', 'g'), ('b', 'y'), ('maroon', 'black'), ('gold', 'pink'), ('purple', 'cyan')]
    est_type_label = ["Lower bound", "Upper bound", "Arithmetic mean", "Geometric mean", "Harmonic mean"]
    for l, n_est_type in enumerate(["lower", "upper", "arith", "geo", "harm"]):
        linear_no_hh, linear_no_hh_exacts = read_sim_data(deg, f"bucket_linear_{n_est_type}_rel_k=k_kh=0", sample_sizes)
        linear_with_hh, linear_with_hh_exacts = read_sim_data(deg, f"bucket_linear_{n_est_type}_rel_k=k/2_kh=k/2", sample_sizes)

        expo_no_hh, expo_no_hh_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_rel_k=k_kh=0", sample_sizes)
        expo_with_hh, expo_with_hh_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_rel_k=k/2_kh=k/2", sample_sizes)
        
        plot_mse(ax[0], sample_sizes, linear_no_hh, linear_no_hh_exacts, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
        plot_mse(ax[0], sample_sizes, linear_with_hh, linear_with_hh_exacts, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])

        plot_mse(ax[1], sample_sizes, expo_no_hh, expo_no_hh_exacts, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
        plot_mse(ax[1], sample_sizes, expo_with_hh, expo_with_hh_exacts, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])


    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    ax[0].set_title("Linear buckets")
    ax[1].set_title("Exponential buckets")

    fig.supxlabel('Space size ($k$)')
    fig.supylabel(f'Normalized Root Mean Squared Error')
    fig.suptitle(f'{nth(deg)} Moment NRMSE vs. Space size, Bucketing Sketch with relative error')

    plt.tight_layout()
    ax[0].get_legend().remove()
    sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    # plt.savefig(f'figs/bucket_sketch_sim_{nth(deg)}_moment_nrmse_expo_rel_err', bbox_inches='tight')
    plt.show()

plot_bucketing_kh_sketches(3)
plot_bucketing_kh_sketches(4)


# %%
def plot_bucketing_kh_sketches(deg):
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 6))

    c = [('r', 'g'), ('b', 'y'), ('gold', 'pink')]
    est_type_label = ["Unbiased estimator", "2 Summary estimator", "Arithmetic mean estimator"]
    oracle_type = 'rel_0.05'
    for l, n_est_type in enumerate(["cond", "alt", "arith"]):
        if n_est_type == "cond":
            linear_with_hh, linear_with_hh_exacts = read_sim_data(deg, f"cond_bucket_{oracle_type}_k=1_kh=k", sample_sizes)
            expo_with_hh, expo_with_hh_exacts = read_sim_data(deg, f"cond_bucket_{oracle_type}_k=1_kh=k", sample_sizes)
            plot_mse(ax[0], sample_sizes, linear_with_hh, linear_with_hh_exacts, f"{est_type_label[l]}", c[l][1])
            plot_mse(ax[1], sample_sizes, expo_with_hh, expo_with_hh_exacts, f"{est_type_label[l]}", c[l][1])
        else:
            linear_no_hh, linear_no_hh_exacts = read_sim_data(deg, f"bucket_linear_{n_est_type}_{oracle_type}_k=k_kh=0", sample_sizes)
            linear_with_hh, linear_with_hh_exacts = read_sim_data(deg, f"bucket_linear_{n_est_type}_{oracle_type}_k=k/2_kh=k/2", sample_sizes)
    
            expo_no_hh, expo_no_hh_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_{oracle_type}_k=k_kh=0", sample_sizes)
            expo_with_hh, expo_with_hh_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_{oracle_type}_k=k/2_kh=k/2", sample_sizes)
        
            plot_mse(ax[0], sample_sizes, linear_no_hh, linear_no_hh_exacts, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
            plot_mse(ax[0], sample_sizes, linear_with_hh, linear_with_hh_exacts, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])
    
            plot_mse(ax[1], sample_sizes, expo_no_hh, expo_no_hh_exacts, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
            plot_mse(ax[1], sample_sizes, expo_with_hh, expo_with_hh_exacts, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])


    ax[0].set_yscale('log')
    ax[0].set_xscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    ax[0].set_title("Linear buckets")
    ax[1].set_title("Exponential buckets")

    fig.supxlabel('Space size ($k$)')
    fig.supylabel(f'Normalized Root Mean Squared Error')
    fig.suptitle(f'{nth(deg)} Moment NRMSE vs. Space size, Bucketing Sketch with relative error')

    plt.tight_layout()
    ax[0].get_legend().remove()
    sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
    # plt.savefig(f'figs/bucket_sketch_new_estimators_{nth(deg)}_moment_nrmse_rel_err', bbox_inches='tight')
    plt.show()

plot_bucketing_kh_sketches(3)
# plot_bucketing_kh_sketches(4)

# %%
# Plot MSE of simulations across error types

sim_folder = "simulation/results"

a = 0.5
deg3_results = pd.read_csv(f"{sim_folder}/new_deg=3_fake_{a}_results.csv")
deg4_results = pd.read_csv(f"{sim_folder}/new_deg=4_fake_{a}_results.csv")
# deg3_results = pd.read_csv(f"{sim_folder}/new_deg=3_aol_results.csv")
# deg4_results = pd.read_csv(f"{sim_folder}/new_deg=4_aol_results.csv")

sample_sizes = [2**i for i in range(6, 17)]
def read_sim_data(deg, sketch_type, sample_sizes):
    df = deg3_results if deg == 3 else deg4_results
    estimates = []
    exacts = []
    for k in sample_sizes:
        mask = (df["sketch_type"] == sketch_type) & (df["k"] == k)
        estimates.append(df.loc[mask, "estimate"].apply(float.fromhex).to_numpy())
        exacts.append(df.loc[mask, "exact"].apply(float).to_numpy())
    return np.array(estimates), np.array(exacts)

def plot_mse_error(deg):
    # ppswor_results, ppswor_exacts = read_sim_data(deg, "ppswor", sample_sizes)
    exact_results, exact_exacts = read_sim_data(deg, "exact", sample_sizes)

    swa_rel_results, swa_rel_exacts = read_sim_data(deg, "swa_rel_0.05_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_train_results, swa_train_exacts = read_sim_data(deg, "swa_train_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_abs_results, swa_abs_exacts = read_sim_data(deg, "swa_abs_0.001_kh=k/2_kp=k/2_ku=0", sample_sizes)

    n_est_type = "alt"
    expo_rel_1, expo_rel_1_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_rel_0.05_k=k/2_kh=k/2", sample_sizes)
    expo_train_1, expo_train_1_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_train_k=k/2_kh=k/2", sample_sizes)
    expo_abs_1, expo_abs_1_exacts = read_sim_data(deg, f"bucket_expo_{n_est_type}_abs_0.001_k=k/2_kh=k/2", sample_sizes)

    cond_rel, cond_rel_exacts = read_sim_data(deg, f"cond_bucket_rel_0.05_k=1_kh=k", sample_sizes)
    cond_train, cond_train_exacts = read_sim_data(deg, f"cond_bucket_train_k=1_kh=k", sample_sizes)
    cond_abs, cond_abs_exacts = read_sim_data(deg, f"cond_bucket_abs_0.001_k=1_kh=k", sample_sizes)

    swa_rel1, swa_rel1_exacts = read_sim_data(deg, f"swa_bucket_expo_rel_0.05_k=64_kp=k/2_kh=k/2", sample_sizes)
    swa_train1, swa_train1_exacts = read_sim_data(deg, f"swa_bucket_expo_train_k=64_kp=k/2_kh=k/2", sample_sizes)
    swa_abs1, swa_abs1_exacts = read_sim_data(deg, f"swa_bucket_expo_abs_0.001_k=64_kp=k/2_kh=k/2", sample_sizes)

    swa_rel2, swa_rel2_exacts = read_sim_data(deg, f"unif_bucket_expo_rel_0.05_k=64_ku=k/2_kh=k/2", sample_sizes)
    swa_train2, swa_train2_exacts = read_sim_data(deg, f"unif_bucket_expo_train_k=64_ku=k/2_kh=k/2", sample_sizes)
    swa_abs2, swa_abs2_exacts = read_sim_data(deg, f"unif_bucket_expo_abs_0.001_k=64_ku=k/2_kh=k/2", sample_sizes)


    fig, ax = plt.subplots()

    plot_mse(ax, sample_sizes, exact_results, exact_exacts, "Exact", 'g')

    plot = plot_mse

    plot(ax, sample_sizes, swa_rel_results, swa_rel_exacts, "SWA: Relative Error,\n$k_h = 0, k_p = k, k_u = 0$", 'r')
    plot(ax, sample_sizes, swa_train_results, swa_train_exacts, "SWA: Train/Test Error,\n$k_h = 0, k_p = k, k_u = 0$", 'cyan')
    plot_mse(ax, sample_sizes, swa_abs_results, swa_abs_exacts, "SWA: Absolute Error,\n$k_h = 0, k_p = k, k_u = 0$", 'b')

    plot(ax, sample_sizes, expo_rel_1, expo_rel_1_exacts, "Bucket: Relative Error,\n$B = k, k_h = 0$", 'pink')
    plot(ax, sample_sizes, expo_train_1, expo_train_1_exacts, "Bucket: Train/Test Error,\n$B = k, k_h = 0$", 'black')
    plot(ax, sample_sizes, expo_abs_1, expo_abs_1_exacts, "Bucket: Absolute Error,\n$B = k, k_h = 0$", 'grey')

    # plot(ax, sample_sizes, cond_rel, cond_rel_exacts, "Unbiased: Relative Error,\n$k_h = 0$", 'gold')
    # plot(ax, sample_sizes, cond_train, cond_train_exacts, "Unbiased: Train/Test Error,\n$k_h = 0$", 'yellow')
    # plot(ax, sample_sizes, cond_abs, cond_abs_exacts, "Unbiased: Absolute Error,\n$k_h = 0$", 'purple')

    # plot(ax, sample_sizes, swa_rel1, swa_rel1_exacts, "SWA Bucket: Relative Error,\n$B = k, k_h = 0$", 'pink')
    # plot(ax, sample_sizes, swa_train1, swa_train1_exacts, "SWA Bucket: Train/Test Error,\n$B = k, k_h = 0$", 'black')
    # plot(ax, sample_sizes, swa_abs1, swa_abs1_exacts, "SWA Bucket: Absolute Error,\n$B = k, k_h = 0$", 'grey')

    # plot(ax, sample_sizes, swa_rel2, swa_rel2_exacts, "Unif Bucket: Relative Error,\n$k_h = 0$", 'gold')
    # plot(ax, sample_sizes, swa_train2, swa_train2_exacts, "Unif Bucket: Train/Test Error,\n$k_h = 0$", 'yellow')
    # plot(ax, sample_sizes, swa_abs2, swa_abs2_exacts, "Unif Bucket: Absolute Error,\n$k_h = 0$", 'purple')

    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')

    plt.xlabel('Sample size ($k$)')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE, SWA vs. Bucketing Sketch')

    # plt.savefig(f'figs/freq_bucket_sketch_sim_nrmse_{nth(deg)}_moment_error_type_focused', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)

# %%
# Plot MSE of simulations across error types

sim_folder = "simulation/results"

a = 0.5
# threshold_results = pd.read_csv(f"{sim_folder}/threshold_fake_{a}_results.csv")
threshold_results = pd.read_csv(f"{sim_folder}/threshold_aol_results.csv")


sample_sizes = [2**i for i in range(6, 13)]
def read_threshold_sim_data(sketch_type, sample_sizes):
    df = threshold_results
    estimates = []
    exacts = []
    for k in sample_sizes:
        mask = (df["sketch_type"] == sketch_type) & (df["k"] == k)
        estimates.append(df.loc[mask, "estimate"].apply(float.fromhex).to_numpy())
        exacts.append(df.loc[mask, "exact"].apply(float).to_numpy())
    return np.array(estimates), np.array(exacts)

def plot_mse_error():
    # ppswor_results, ppswor_exacts = read_sim_data(deg, "ppswor", sample_sizes)
    exact_results, exact_exacts = read_threshold_sim_data("exact", sample_sizes)

    swa_rel_results, swa_rel_exacts = read_threshold_sim_data("swa_rel_0.05_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_train_results, swa_train_exacts = read_threshold_sim_data("swa_train_kh=k/2_kp=k/2_ku=0", sample_sizes)
    swa_abs_results, swa_abs_exacts = read_threshold_sim_data("swa_abs_0.001_kh=k/2_kp=k/2_ku=0", sample_sizes)

    n_est_type = "arith"
    expo_rel_1, expo_rel_1_exacts = read_threshold_sim_data(f"bucket_expo_{n_est_type}_rel_0.05_k=k/2_kh=k/2", sample_sizes)
    expo_train_1, expo_train_1_exacts = read_threshold_sim_data(f"bucket_expo_{n_est_type}_train_k=k/2_kh=k/2", sample_sizes)
    expo_abs_1, expo_abs_1_exacts = read_threshold_sim_data(f"bucket_expo_{n_est_type}_abs_0.001_k=k/2_kh=k/2", sample_sizes)

    # cond_rel, cond_rel_exacts = read_sim_data(deg, f"cond_bucket_rel_0.05_k=1_kh=k", sample_sizes)
    # cond_train, cond_train_exacts = read_sim_data(deg, f"cond_bucket_train_k=1_kh=k", sample_sizes)
    # cond_abs, cond_abs_exacts = read_sim_data(deg, f"cond_bucket_abs_0.001_k=1_kh=k", sample_sizes)

    swa_rel1, swa_rel1_exacts = read_threshold_sim_data(f"swa_bucket_expo_rel_0.05_k=64_kp=k/2_kh=k/2", sample_sizes)
    swa_train1, swa_train1_exacts = read_threshold_sim_data(f"swa_bucket_expo_train_k=64_kp=k/2_kh=k/2", sample_sizes)
    swa_abs1, swa_abs1_exacts = read_threshold_sim_data(f"swa_bucket_expo_abs_0.001_k=64_kp=k/2_kh=k/2", sample_sizes)

    swa_rel2, swa_rel2_exacts = read_threshold_sim_data(f"unif_bucket_expo_rel_0.05_k=64_ku=k/2_kh=k/2", sample_sizes)
    swa_train2, swa_train2_exacts = read_threshold_sim_data(f"unif_bucket_expo_train_k=64_ku=k/2_kh=k/2", sample_sizes)
    swa_abs2, swa_abs2_exacts = read_threshold_sim_data(f"unif_bucket_expo_abs_0.001_k=64_ku=k/2_kh=k/2", sample_sizes)


    fig, ax = plt.subplots()

    plot_mse(ax, sample_sizes, exact_results, exact_exacts, "Exact", 'g')

    plot = plot_mse

    plot(ax, sample_sizes, swa_rel_results, swa_rel_exacts, "SWA: Relative Error,\n$k_h = 0, k_p = k, k_u = 0$", 'r')
    plot(ax, sample_sizes, swa_train_results, swa_train_exacts, "SWA: Train/Test Error,\n$k_h = 0, k_p = k, k_u = 0$", 'cyan')
    plot_mse(ax, sample_sizes, swa_abs_results, swa_abs_exacts, "SWA: Absolute Error,\n$k_h = 0, k_p = k, k_u = 0$", 'b')

    plot(ax, sample_sizes, expo_rel_1, expo_rel_1_exacts, "Bucket: Relative Error,\n$B = k, k_h = 0$", 'pink')
    plot(ax, sample_sizes, expo_train_1, expo_train_1_exacts, "Bucket: Train/Test Error,\n$B = k, k_h = 0$", 'black')
    plot(ax, sample_sizes, expo_abs_1, expo_abs_1_exacts, "Bucket: Absolute Error,\n$B = k, k_h = 0$", 'grey')

    # plot(ax, sample_sizes, cond_rel, cond_rel_exacts, "Unbiased: Relative Error,\n$k_h = 0$", 'gold')
    # plot(ax, sample_sizes, cond_train, cond_train_exacts, "Unbiased: Train/Test Error,\n$k_h = 0$", 'yellow')
    # plot(ax, sample_sizes, cond_abs, cond_abs_exacts, "Unbiased: Absolute Error,\n$k_h = 0$", 'purple')

    # plot(ax, sample_sizes, swa_rel1, swa_rel1_exacts, "SWA Bucket: Relative Error,\n$B = k, k_h = 0$", 'pink')
    # plot(ax, sample_sizes, swa_train1, swa_train1_exacts, "SWA Bucket: Train/Test Error,\n$B = k, k_h = 0$", 'black')
    # plot(ax, sample_sizes, swa_abs1, swa_abs1_exacts, "SWA Bucket: Absolute Error,\n$B = k, k_h = 0$", 'grey')

    # plot(ax, sample_sizes, swa_rel2, swa_rel2_exacts, "Unif Bucket: Relative Error,\n$k_h = 0$", 'gold')
    # plot(ax, sample_sizes, swa_train2, swa_train2_exacts, "Unif Bucket: Train/Test Error,\n$k_h = 0$", 'yellow')
    # plot(ax, sample_sizes, swa_abs2, swa_abs2_exacts, "Unif Bucket: Absolute Error,\n$k_h = 0$", 'purple')

    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')

    plt.xlabel('Sample size ($k$)')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'Threshold NRMSE, SWA vs. Bucketing Sketch')

    # plt.savefig(f'figs/freq_bucket_sketch_sim_nrmse_{nth(deg)}_moment_error_type_focused', bbox_inches='tight')
    plt.show()

plot_mse_error()
