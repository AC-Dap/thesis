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
df = pd.read_csv('data/AOL-user-ct-collection/user-ct-test-collection-01.txt', sep='\t')
df['Query'] = df['Query'].fillna("")
unique_counts = df['Query'].value_counts()
df.head()

# %%
plt.plot(np.arange(len(unique_counts)), unique_counts)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Rank vs. Frequency of search results')
# plt.savefig('figs/aol_rank_frequency')

# %%
sim_folder = "simulation/results"

nsims = 30
sample_sizes = [2**i for i in range(8, 17)]
def read_sim_data(path):
    with open(f"{sim_folder}/{path}", 'r') as f:
        lines = [line.rstrip() for line in f]
        return [float(lines[i]) for i in range(nsims)]

def get_exact_moment(deg):
    return float(sum([val**deg for val in unique_counts]))

def nth(deg):
    return "3rd" if deg == 3 else f"{deg}th"

def error_prefix(err):
    if err == "relative": return "rel"
    if err == "absolute": return "abs"
    if err == "binomial": return "bin"
    if err == "test": return "test"


# %%
def plot_curve(ax, x, results, true_value, label, color):
    results = np.array(results)
    error = (results - true_value) / true_value
    mean = np.mean(error, axis=1)
    lower = np.min(error, axis=1)
    upper = np.max(error, axis=1)
    
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, lower, upper, color=color, alpha=0.2)
    ax.legend()

def plot_abs_curve(ax, x, results, true_value, label, color):
    results = np.array(results)
    error = (results - true_value) / true_value
    mean = np.mean(np.abs(error), axis=1)
    upper = np.max(np.abs(error), axis=1)
    
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean, upper, color=color, alpha=0.2)
    ax.legend()

def plot_mse(ax, x, results, true_value, label, color):
    results = np.array(results)
    mse = np.sqrt(np.mean((results - true_value)**2, axis=1)) / true_value
    
    ax.plot(x, mse, label=label, color=color)


# %%
# Plot absolute estimation error vs. sample size

def plot_error_sample_size(deg, error_type):
    ep = 0.05
    
    true_value = get_exact_moment(deg)
    print(true_value)

    prefix = error_prefix(error_type)
    
    ppswor_results = [read_sim_data(f"ppswor_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_1_results = [read_sim_data(f"swa_k=0-{k}-0_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_2_results = [read_sim_data(f"swa_k=0-{k//2}-{k//2}_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_3_results = [read_sim_data(f"swa_k=100-{k}-0_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_4_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    
    fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(12, 5))
    
    plot_curve(ax[0], sample_sizes, ppswor_results, true_value, "PPSWOR", 'b')
    
    plot_curve(ax[1], sample_sizes, swa_1_results, true_value, "SWA: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_curve(ax[2], sample_sizes, swa_2_results, true_value, "SWA: $k_h = 0$,\n$k_u = k_p$", 'r')
    plot_curve(ax[3], sample_sizes, swa_3_results, true_value, "SWA: $k_h = 100$,\n$k_u = 0$", 'g')
    plot_curve(ax[4], sample_sizes, swa_4_results, true_value, "SWA: $k_h = 100$,\n$k_u = k_p$", 'y')
    
    # ax[0].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    # ax[1].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    
    ax[0].set_xscale('log')
    
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
    
    true_value = get_exact_moment(deg)
    print(true_value)

    prefix = error_prefix(error_type)
    
    ppswor_results = [read_sim_data(f"ppswor_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_1_results = [read_sim_data(f"swa_k=0-{k}-0_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_2_results = [read_sim_data(f"swa_k=0-{k//2}-{k//2}_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_3_results = [read_sim_data(f"swa_k=100-{k}-0_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_4_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_{prefix}_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    
    fig, ax = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0], sample_sizes, ppswor_results, true_value, "PPSWOR", 'b')
    
    plot_abs_curve(ax[1], sample_sizes, swa_1_results, true_value, "SWA: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_abs_curve(ax[2], sample_sizes, swa_2_results, true_value, "SWA: $k_h = 0$,\n$k_u = k_p$", 'r')
    plot_abs_curve(ax[3], sample_sizes, swa_3_results, true_value, "SWA: $k_h = 100$,\n$k_u = 0$", 'g')
    plot_abs_curve(ax[4], sample_sizes, swa_4_results, true_value, "SWA: $k_h = 100$,\n$k_u = k_p$", 'y')
    
    # ax[0].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    # ax[1].axhline(actual_value, label="L2 Norm", c='black', linestyle='dashed')
    
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    
    fig.supxlabel('Sample size')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Absolute Estimation Error vs. Sample Size, using {ep} {error_type} error')
    
    plt.savefig(f'figs/freq_sketch_sim_{nth(deg)}_moment_absolute_{prefix}_err')
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
    
    true_value = get_exact_moment(deg)
    print(true_value)
    
    swa_rel_1_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_abs_1_results = [read_sim_data(f"swa_k=100-{k}-0_abs_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_abs_2_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_abs_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    
    fig, ax = plt.subplots(2, 2,  sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0][0], sample_sizes, swa_rel_1_results, true_value, "SWA Rel: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_abs_curve(ax[0][1], sample_sizes, swa_rel_2_results, true_value, "SWA Rel: $k_h = 0$,\n$k_u = k_p$", 'r')
    plot_abs_curve(ax[1][0], sample_sizes, swa_abs_1_results, true_value, "SWA Abs: $k_h = 0$,\n$k_u = 0$", 'g')
    plot_abs_curve(ax[1][1], sample_sizes, swa_abs_2_results, true_value, "SWA Abs: $k_h = 0$,\n$k_u = k_p$", 'y')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')
    
    fig.supxlabel('Sample size')
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
    
    true_value = get_exact_moment(deg)
    print(true_value)
    
    exact_results = [read_sim_data(f"exact_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_abs_results = [read_sim_data(f"swa_k=100-{k}-0_abs_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_bin_results = [read_sim_data(f"swa_k=100-{k}-0_bin_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    
    fig, ax = plt.subplots(2, 2,  sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0][0], sample_sizes, exact_results, true_value, "Exact: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_abs_curve(ax[0][1], sample_sizes, swa_rel_results, true_value, "SWA Rel: $k_h = 0$,\n$k_u = 0$", 'r')
    plot_abs_curve(ax[1][0], sample_sizes, swa_abs_results, true_value, "SWA Abs: $k_h = 0$,\n$k_u = 0$", 'g')
    plot_abs_curve(ax[1][1], sample_sizes, swa_bin_results, true_value, "SWA Bin: $k_h = 0$,\n$k_u = 0$", 'y')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')
    
    fig.supxlabel('Sample size')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Absolute Estimation Error vs. Error Type')
    
    plt.savefig(f'figs/freq_sketch_sim_{nth(deg)}_moment_error_type')
    plt.show()

plot_error_oracle_types(3)
plot_error_oracle_types(4)


# %%
# Plot MSE of simulations across epsilons

def plot_mse_rel(deg):
    sample_sizes = [256, 1024, 4096, 16384, 65536]
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    ppswor_results = [read_sim_data(f"ppswor_k={k}_deg={deg}.txt") for k in sample_sizes]
    exact_results = [read_sim_data(f"exact_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_1_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep=0.05_deg={deg}.txt") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_rel_ep=0.05_deg={deg}.txt") for k in sample_sizes]
    swa_rel_3_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep={0.2}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_4_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_rel_ep={0.2}_deg={deg}.txt") for k in sample_sizes]

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, ppswor_results, true_value, "PPSWOR", 'r')
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_1_results, true_value, "SWA ep=0.05: $k_h = 100$,\n$k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_2_results, true_value, "SWA ep=0.05: $k_h = 100$,\n$k_u = k_p$", 'y')
    plot_mse(ax, sample_sizes, swa_rel_3_results, true_value, "SWA ep=0.2: $k_h = 100$,\n$k_u = 0$", 'orange')
    plot_mse(ax, sample_sizes, swa_rel_4_results, true_value, "SWA ep=0.2: $k_h = 100$,\n$k_u = k_p$", 'purple')
    
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Sample size')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Epsilon')

    plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_epsilon', bbox_inches='tight')
    plt.show()

plot_mse_rel(3)
plot_mse_rel(4)


# %%
# Plot MSE of simulations across error types

def plot_mse_error(deg):
    ep = 0.05
    sample_sizes = [256, 1024, 4096, 16384, 65536]
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    ppswor_results = [read_sim_data(f"ppswor_k={k}_deg={deg}.txt") for k in sample_sizes]
    exact_results = [read_sim_data(f"exact_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_1_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_abs_1_results = [read_sim_data(f"swa_k=100-{k}-0_abs_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_abs_2_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_abs_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_bin_1_results = [read_sim_data(f"swa_k=100-{k}-0_bin_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_bin_2_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_bin_ep={ep}_deg={deg}.txt") for k in sample_sizes]

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, ppswor_results, true_value, "PPSWOR", 'r')
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_1_results, true_value, "SWA Rel: $k_h = 100$,\n$k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_abs_1_results, true_value, "SWA Abs: $k_h = 100$,\n$k_u = 0$", 'black')
    plot_mse(ax, sample_sizes, swa_bin_1_results, true_value, "SWA Bin: $k_h = 100$,\n$k_u = 0$", 'grey')
    plot_mse(ax, sample_sizes, swa_rel_2_results, true_value, "SWA Rel: $k_h = 100$,\n$k_u = k_p$", 'y')
    plot_mse(ax, sample_sizes, swa_abs_2_results, true_value, "SWA Abs: $k_h = 100$,\n$k_u = k_p$", 'purple')
    plot_mse(ax, sample_sizes, swa_bin_2_results, true_value, "SWA Bin: $k_h = 100$,\n$k_u = k_p$", 'pink')
    
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Sample size')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Error Type')

    plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_error_type', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)


# %%
# Plot MSE of simulations across error types

def plot_mse_error(deg):
    ep = 0.05
    sample_sizes = [256, 1024, 4096, 16384, 65536]
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    exact_results = [read_sim_data(f"exact_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_abs_results = [read_sim_data(f"swa_k=100-{k}-0_abs_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_bin_results = [read_sim_data(f"swa_k=100-{k}-0_bin_ep={ep}_deg={deg}.txt") for k in sample_sizes]

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_results, true_value, "SWA Rel: $k_h = 100$,\n$k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_abs_results, true_value, "SWA Abs: $k_h = 100$,\n$k_u = 0$", 'r')
    plot_mse(ax, sample_sizes, swa_bin_results, true_value, "SWA Bin: $k_h = 100$,\n$k_u = 0$", 'y')

    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Sample size')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Error Type')

    plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_error_type_focused', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)


# %%
# Plot MSE of simulations across kh

def plot_curve(x, results, true_value, label, color):
    results = np.array(results)
    mse = np.sqrt(np.mean((results - true_value)**2, axis=1)) / true_value
    
    plt.plot(x, mse, label=label, color=color)

def plot_mse_kh(deg):
    ep = 0.05
    sample_sizes = [256, 1024, 4096, 16384, 65536]
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    ppswor_results = [read_sim_data(f"ppswor_k={k}_deg={deg}.txt") for k in sample_sizes]
    exact_results = [read_sim_data(f"exact_k={k}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_1_results = [read_sim_data(f"swa_k=0-{k}-0_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(f"swa_k=0-{k//2}-{k//2}_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_3_results = [read_sim_data(f"swa_k=100-{k}-0_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]
    swa_rel_4_results = [read_sim_data(f"swa_k=100-{k//2}-{k//2}_rel_ep={ep}_deg={deg}.txt") for k in sample_sizes]

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, ppswor_results, true_value, "PPSWOR", 'r')
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_1_results, true_value, "SWA: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_1_results, true_value, "SWA: $k_h = 100$,\n$k_u = 0$", 'black')
    plot_mse(ax, sample_sizes, swa_rel_2_results, true_value, "SWA: $k_h = 0$,\n$k_u = k_p$", 'y')
    plot_mse(ax, sample_sizes, swa_rel_2_results, true_value, "SWA: $k_h = 100$,\n$k_u = k_p$", 'cyan')
    
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('Sample size')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. $k_h$, relative error')

    plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_kh', bbox_inches='tight')
    plt.show()

plot_mse_kh(3)
plot_mse_kh(4)
