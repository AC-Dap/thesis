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
plt.ylabel('Weight')
plt.xscale('log')
plt.yscale('log')
plt.title('Rank vs. Weight of search results')
plt.savefig('figs/aol_rank_weight')

# %%
plt.plot(np.arange(len(unique_counts)), unique_counts / len(df))
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Rank vs. Frequency of search results')
plt.savefig('figs/aol_rank_frequency')

# %%
sim_folder = "simulation/results"

nsims = 30
sample_sizes = [2**i for i in range(6, 17)]
def read_sim_data(deg, algo, params):
    with open(f"{sim_folder}/deg={deg}/{algo}/{algo}_{params}.txt", 'r') as f:
        lines = [line.rstrip() for line in f]
        return [float.fromhex(lines[i]) for i in range(min(nsims, len(lines)))]

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

def plot_mse(ax, x, results, true_value, label, color, linestyle='solid'):
    results = np.array(results)
    mse = np.sqrt(np.mean((results - true_value)**2, axis=1)) / true_value
    
    ax.plot(x, mse, label=label, color=color, linestyle=linestyle)
    ax.legend()


# %%
# Plot absolute estimation error vs. sample size

def plot_error_sample_size(deg, error_type):
    ep = 0.05
    
    true_value = get_exact_moment(deg)
    print(true_value)

    prefix = error_prefix(error_type)
    
    ppswor_results = [read_sim_data(deg, "ppswor", f"k={k}_deg={deg}") for k in sample_sizes]
    swa_1_results = [read_sim_data(deg, f"swa_{prefix}", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_2_results = [read_sim_data(deg, f"swa_{prefix}", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_3_results = [read_sim_data(deg, f"swa_{prefix}", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 5))
    
    plot_curve(ax[0], sample_sizes, ppswor_results, true_value, "PPSWOR", 'b')
    plot_curve(ax[1], sample_sizes, swa_1_results, true_value, "SWA: $k_h = 0$,\n$k_u = 0$", 'b')
    plot_curve(ax[2], sample_sizes, swa_2_results, true_value, "SWA: $k_h = 0$,\n$k_u = k_p$", 'r')
    plot_curve(ax[3], sample_sizes, swa_3_results, true_value, "SWA: $k_h = k_p$,\n$k_u = 0$", 'g')
    
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
    
    true_value = get_exact_moment(deg)
    print(true_value)

    prefix = error_prefix(error_type)
    
    ppswor_results = [read_sim_data(deg, "ppswor", f"k={k}_deg={deg}") for k in sample_sizes]
    swa_1_results = [read_sim_data(deg, f"swa_{prefix}", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_2_results = [read_sim_data(deg, f"swa_{prefix}", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_3_results = [read_sim_data(deg, f"swa_{prefix}", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0], sample_sizes, ppswor_results, true_value, "PPSWOR", 'b')
    plot_abs_curve(ax[1], sample_sizes, swa_1_results, true_value, "SWA: $k_h = 0$,\n$k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[2], sample_sizes, swa_2_results, true_value, "SWA: $k_h = 0$,\n$k_p = \\frac{k}{2}, k_u= \\frac{k}{2}$", 'r')
    plot_abs_curve(ax[3], sample_sizes, swa_3_results, true_value, "SWA: $k_h = \\frac{k}{2}$,\n$k_p = \\frac{k}{2}, k_u = 0$", 'g')
    
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
    
    swa_rel_1_results = [read_sim_data(deg, "swa_rel", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(deg, "swa_rel", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_3_results = [read_sim_data(deg, "swa_rel", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    swa_abs_1_results = [read_sim_data(deg, "swa_abs", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_abs_2_results = [read_sim_data(deg, "swa_abs", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_abs_3_results = [read_sim_data(deg, "swa_abs", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    swa_bin_1_results = [read_sim_data(deg, "swa_bin", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_2_results = [read_sim_data(deg, "swa_bin", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_3_results = [read_sim_data(deg, "swa_bin", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    
    fig, ax = plt.subplots(3, 3,  sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0][0], sample_sizes, swa_rel_1_results, true_value, "Relative Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[0][1], sample_sizes, swa_rel_2_results, true_value, "Relative Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'b')
    plot_abs_curve(ax[0][2], sample_sizes, swa_rel_3_results, true_value, "Relative Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')

    plot_abs_curve(ax[1][0], sample_sizes, swa_abs_1_results, true_value, "Absolute Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[1][1], sample_sizes, swa_abs_2_results, true_value, "Absolute Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'b')
    plot_abs_curve(ax[1][2], sample_sizes, swa_abs_3_results, true_value, "Absolute Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')

    plot_abs_curve(ax[2][0], sample_sizes, swa_bin_1_results, true_value, "Binomial Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_abs_curve(ax[2][1], sample_sizes, swa_bin_2_results, true_value, "Binomial Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'b')
    plot_abs_curve(ax[2][2], sample_sizes, swa_bin_3_results, true_value, "Binomial Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')

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
    
    true_value = get_exact_moment(deg)
    print(true_value)
    
    exact_results = [read_sim_data(deg, "exact", f"k={k}_deg={deg}") for k in sample_sizes]
    swa_rel_results = [read_sim_data(deg, "swa_rel", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_abs_results = [read_sim_data(deg, "swa_abs", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_results = [read_sim_data(deg, "swa_bin", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    
    fig, ax = plt.subplots(2, 2,  sharex=True, sharey=True, figsize=(12, 5))
    
    plot_abs_curve(ax[0][0], sample_sizes, exact_results, true_value, "Exact", 'b')
    plot_abs_curve(ax[0][1], sample_sizes, swa_rel_results, true_value, "Relative: $k_h = 0$,\n$k_p = k, k_u = 0$", 'r')
    plot_abs_curve(ax[1][0], sample_sizes, swa_abs_results, true_value, "Absolute: $k_h = 0$,\n$k_p = k, k_u = 0$", 'g')
    plot_abs_curve(ax[1][1], sample_sizes, swa_bin_results, true_value, "Binomial: $k_h = 0$,\n$k_p = k, k_u = 0$", 'y')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')

    ax[1][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    fig.supxlabel('Sample size ($k$)')
    fig.supylabel(f'Absolute Normalized Estimation Error')
    fig.suptitle(f'{nth(deg)} Moment Absolute Estimation Error vs. Error Type')

    plt.tight_layout()
    plt.savefig(f'figs/freq_sketch_sim_{nth(deg)}_moment_error_type')
    plt.show()

plot_error_oracle_types(3)
plot_error_oracle_types(4)


# %%
# Plot MSE of simulations across epsilons

def plot_mse_rel(deg):    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    ppswor_results = [read_sim_data(deg, "ppswor", f"k={k}_deg={deg}") for k in sample_sizes]
    exact_results = [read_sim_data("exact", deg, f"k={k}_deg={deg}") for k in sample_sizes]
    
    swa_rel_1_results = [read_sim_data(deg, "swa_rel", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(deg, "swa_rel", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_3_results = [read_sim_data(deg, "swa_rel", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

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
    ep = 0.05
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    ppswor_results = [read_sim_data(deg, "ppswor", f"k={k}_deg={deg}") for k in sample_sizes]
    exact_results = [read_sim_data(deg, "exact", f"k={k}_deg={deg}") for k in sample_sizes]
    
    swa_rel_1_results = [read_sim_data(deg, "swa_rel", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(deg, "swa_rel", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_3_results = [read_sim_data(deg, "swa_rel", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    swa_abs_1_results = [read_sim_data(deg, "swa_abs", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_abs_2_results = [read_sim_data(deg, "swa_abs", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_abs_3_results = [read_sim_data(deg, "swa_abs", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    swa_bin_1_results = [read_sim_data(deg, "swa_bin", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_2_results = [read_sim_data(deg, "swa_bin", f"k=0-{k//2}-{k//2}_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_3_results = [read_sim_data(deg, "swa_bin", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, ppswor_results, true_value, "PPSWOR", 'r')
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')

    plot_mse(ax, sample_sizes, swa_rel_1_results, true_value, "Relative Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_2_results, true_value, "Relative Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'black')
    plot_mse(ax, sample_sizes, swa_rel_3_results, true_value, "Relative Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'grey')

    # plot_mse(ax, sample_sizes, swa_abs_1_results, true_value, "Absolute Error: \n$k_h = 0, k_p = k, k_u = 0$", 'y')
    # plot_mse(ax, sample_sizes, swa_abs_2_results, true_value, "Absolute Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'purple')
    # plot_mse(ax, sample_sizes, swa_abs_3_results, true_value, "Absolute Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'pink')

    plot_mse(ax, sample_sizes, swa_bin_1_results, true_value, "Binomial Error: \n$k_h = 0, k_p = k, k_u = 0$", 'cyan')
    plot_mse(ax, sample_sizes, swa_bin_2_results, true_value, "Binomial Error: \n$k_h = 0, k_p = \\frac{k}{2}, k_u = \\frac{k}{2}$", 'gold')
    plot_mse(ax, sample_sizes, swa_bin_3_results, true_value, "Binomial Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'orange')

    
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')
    
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
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    exact_results = [read_sim_data(deg, "exact", f"k={k}_deg={deg}") for k in sample_sizes]
    
    swa_rel_1_results = [read_sim_data(deg, "swa_rel", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_rel_2_results = [read_sim_data(deg, "swa_rel", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_1_results = [read_sim_data(deg, "swa_bin", f"k=0-{k}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_2_results = [read_sim_data(deg, "swa_bin", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')
    plot_mse(ax, sample_sizes, swa_rel_1_results, true_value, "Relative Error: \n$k_h = 0, k_p = k, k_u = 0$", 'b')
    plot_mse(ax, sample_sizes, swa_rel_2_results, true_value, "Relative Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'r')
    plot_mse(ax, sample_sizes, swa_bin_1_results, true_value, "Binomial Error: \n$k_h = 0, k_p = k, k_u = 0$", 'grey')
    plot_mse(ax, sample_sizes, swa_bin_2_results, true_value, "Binomial Error: \n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'cyan')

    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.xscale('log')

    plt.xticks(sample_sizes, sample_sizes, rotation='vertical')
    
    plt.xlabel('Sample size ($k$)')
    plt.ylabel(f'Normalized Root Mean Squared Error')
    plt.title(f'{nth(deg)} Moment NRMSE vs. Error Type')

    plt.savefig(f'figs/freq_sketch_sim_nrmse_{nth(deg)}_moment_error_type_focused', bbox_inches='tight')
    plt.show()

plot_mse_error(3)
plot_mse_error(4)


# %%
# Plot MSE of simulations across error types

def plot_mse_error(deg):
    ep = 0.05
    
    true_value = float(sum([val**deg for val in unique_counts]))
    print(true_value)
    
    exact_results = [read_sim_data(deg, "exact", f"k={k}_deg={deg}") for k in sample_sizes]
    
    swa_rel_results = [read_sim_data(deg, "swa_rel", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_bin_results = [read_sim_data(deg, "swa_bin", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]
    swa_abs_results = [read_sim_data(deg, "swa_abs", f"k={k//2}-{k//2}-0_ep={ep}_deg={deg}") for k in sample_sizes]

    n_est_type = "harm"
    expo_rel_1 = [read_sim_data(deg, f"expo_bucket_{n_est_type}_rel", f"k={k//2}_khh={k//2}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
    expo_bin_1 = [read_sim_data(deg, f"expo_bucket_{n_est_type}_bin", f"k={k//2}_khh={k//2}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
    expo_abs_1 = [read_sim_data(deg, f"expo_bucket_{n_est_type}_abs", f"k={k//2}_khh={k//2}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]

    expo_rel_2 = [read_sim_data(deg, f"expo_bucket_{n_est_type}_rel", f"k={k}_khh={0}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
    expo_bin_2 = [read_sim_data(deg, f"expo_bucket_{n_est_type}_bin", f"k={k}_khh={0}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
    expo_abs_2 = [read_sim_data(deg, f"expo_bucket_{n_est_type}_abs", f"k={k}_khh={0}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
    
    fig, ax = plt.subplots()
    
    plot_mse(ax, sample_sizes, exact_results, true_value, "Exact", 'g')
    
    plot_mse(ax, sample_sizes, swa_rel_results, true_value, "SWA: Relative Error,\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'r')
    plot_mse(ax, sample_sizes, swa_bin_results, true_value, "SWA: Binomial Error,\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'cyan')
    plot_mse(ax, sample_sizes, swa_abs_results, true_value, "SWA: Absolute Error,\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$", 'b')


    plot_mse(ax, sample_sizes, expo_rel_1, true_value, "Bucket: Relative Error,\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'gold')
    plot_mse(ax, sample_sizes, expo_bin_1, true_value, "Bucket: Binomial Error,\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'darkkhaki')
    plot_mse(ax, sample_sizes, expo_abs_1, true_value, "Bucket: Absolute Error,\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$", 'purple')

    plot_mse(ax, sample_sizes, expo_rel_2, true_value, "Bucket: Relative Error,\n$B = k, k_h = 0$", 'pink')
    plot_mse(ax, sample_sizes, expo_bin_2, true_value, "Bucket: Binomial Error,\n$B = k, k_h = 0$", 'brown')
    plot_mse(ax, sample_sizes, expo_abs_2, true_value, "Bucket: Absolute Error,\n$B = k, k_h = 0$", 'grey')

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
    ep = 0.05
    true_value = float(sum([val**deg for val in unique_counts]))

    fig, ax = plt.subplots(2, 3,  sharex=True, sharey=True, figsize=(12, 5))

    for i, bucket_type in enumerate(["linear", "expo"]):
        for j, oracle_type in enumerate(["rel", "abs", "bin"]):
            c = [('r', 'g'), ('b', 'y'), ('maroon', 'black'), ('gold', 'pink'), ('purple', 'cyan')]
            est_type_label = ["Lower bound", "Upper bound", "Arithmetic mean", "Geometric mean", "Harmonic mean"]
            for l, n_est_type in enumerate(["lower", "upper", "arith", "geo", "harm"]):
                param_no_hh = lambda k, ep, deg: f"k={k}_khh={0}_min_freq=7_ep={ep}_deg={deg}" if i == 1 else f"k={k}_khh={0}_ep={ep}_deg={deg}"
                param_with_hh = lambda k, ep, deg: f"k={k//2}_khh={k//2}_min_freq=7_ep={ep}_deg={deg}" if i == 1 else f"k={k//2}_khh={k//2}_ep={ep}_deg={deg}"
                no_hh = [read_sim_data(deg, f"{bucket_type}_bucket_{n_est_type}_{oracle_type}", param_no_hh(k, ep, deg)) for k in sample_sizes]
                with_hh = [read_sim_data(deg, f"{bucket_type}_bucket_{n_est_type}_{oracle_type}", param_with_hh(k, ep, deg)) for k in sample_sizes]

                plot_abs_curve(ax[i][j], sample_sizes, no_hh, true_value, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
                plot_abs_curve(ax[i][j], sample_sizes, with_hh, true_value, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])
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

    plt.savefig(f'figs/bucket_sketch_sim_{nth(deg)}_moment_absolute_err', bbox_inches='tight')
    plt.show()

plot_bucketing_sketches(3)
plot_bucketing_sketches(4)


# %%
def plot_bucketing_kh_sketches(deg):
    ep = 0.05
    true_value = float(sum([val**deg for val in unique_counts]))

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 6))

    c = [('r', 'g'), ('b', 'y'), ('maroon', 'black'), ('gold', 'pink'), ('purple', 'cyan')]
    est_type_label = ["Lower bound", "Upper bound", "Arithmetic mean", "Geometric mean", "Harmonic mean"]
    for l, n_est_type in enumerate(["lower", "upper", "arith", "geo", "harm"]):
        linear_no_hh = [read_sim_data(deg, f"linear_bucket_{n_est_type}_rel", f"k={k}_khh={0}_ep={ep}_deg={deg}") for k in sample_sizes]
        linear_with_hh = [read_sim_data(deg, f"linear_bucket_{n_est_type}_rel", f"k={k//2}_khh={k//2}_ep={ep}_deg={deg}") for k in sample_sizes]

        expo_no_hh = [read_sim_data(deg, f"expo_bucket_{n_est_type}_rel", f"k={k}_khh={0}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
        expo_with_hh = [read_sim_data(deg, f"expo_bucket_{n_est_type}_rel", f"k={k//2}_khh={k//2}_min_freq=7_ep={ep}_deg={deg}") for k in sample_sizes]
        
        plot_mse(ax[0], sample_sizes, linear_no_hh, true_value, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
        plot_mse(ax[0], sample_sizes, linear_with_hh, true_value, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])

        plot_mse(ax[1], sample_sizes, expo_no_hh, true_value, f"{est_type_label[l]}:\n$B=k, k_h = 0$", c[l][0])
        plot_mse(ax[1], sample_sizes, expo_with_hh, true_value, f"{est_type_label[l]}:\n$B=\\frac{{k}}{{2}}, k_h = \\frac{{k}}{{2}}$", c[l][1])


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
    plt.savefig(f'figs/bucket_sketch_sim_{nth(deg)}_moment_nrmse_expo_rel_err', bbox_inches='tight')
    plt.show()

plot_bucketing_kh_sketches(3)
plot_bucketing_kh_sketches(4)
