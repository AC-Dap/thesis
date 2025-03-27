# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# -

df = pd.read_csv("data/processed/AOL/test.txt", names=['id'])
unique_counts = df['id'].value_counts()
unique_freqs = unique_counts / len(df)
prop_of_total = unique_counts ** 3 / np.sum(unique_counts**3)
plt.plot(np.arange(len(prop_of_total)), prop_of_total)
plt.xscale('log')
plt.yscale('log')

# +
sim_folder = "simulation/results"

deg3_aol, deg4_aol = pd.read_csv(f"{sim_folder}/aol_moments_deg=3.csv"), pd.read_csv(f"{sim_folder}/aol_moments_deg=4.csv")
deg3_0_1, deg4_0_1 = pd.read_csv(f"{sim_folder}/fake_0.1_moments_deg=3.csv"), pd.read_csv(f"{sim_folder}/fake_0.1_moments_deg=4.csv")
deg3_0_3, deg4_0_3 = pd.read_csv(f"{sim_folder}/fake_0.3_moments_deg=3.csv"), pd.read_csv(f"{sim_folder}/fake_0.3_moments_deg=4.csv")
deg3_0_5, deg4_0_5 = pd.read_csv(f"{sim_folder}/fake_0.5_moments_deg=3.csv"), pd.read_csv(f"{sim_folder}/fake_0.5_moments_deg=4.csv")
deg3_caida, deg4_caida = pd.read_csv(f"{sim_folder}/caida_moments_deg=3.csv"), pd.read_csv(f"{sim_folder}/caida_moments_deg=4.csv")

threshold_aol = pd.read_csv(f"{sim_folder}/aol_threshold.csv")
threshold_0_1 = pd.read_csv(f"{sim_folder}/fake_0.1_threshold.csv")
threshold_0_3 = pd.read_csv(f"{sim_folder}/fake_0.3_threshold.csv")
threshold_0_5 = pd.read_csv(f"{sim_folder}/fake_0.5_threshold.csv")
threshold_caida = pd.read_csv(f"{sim_folder}/caida_threshold.csv")


# +
def read_sim_data(df, sketch_type):
    estimates = []
    exacts = []
    sample_sizes = df["k"].unique()
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


# +
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

    markers = ['o', 's', '^', 'd']
    marker = markers[hash(label) % len(markers)]
    sns.lineplot(ax=ax, x=x, y=mse, linestyle="dashed", marker=marker, label=label, legend=False)
    # ax.errorbar(x, mse, y_err=(mse - 
    # ax.plot(x, mse, linestyle="dashed", marker="o", label=label)


# +
# Bucketing schemes

def plot_bucket_schemes(df, sample_sizes, title, file_name):
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 5))
    error_types = ["rel_0.05", "abs_0.001", "train"]
    bucket_types = ["linear", "expo"]

    for i, bucket_type in enumerate(bucket_types):
        for j, error_type in enumerate(error_types):
            central_results1, central_exacts1 = read_sim_data(df, f"central_bucket_{bucket_type}_{error_type}_k=k_kh=0")
            central_results2, central_exacts2 = read_sim_data(df, f"central_bucket_{bucket_type}_{error_type}_k=k/2_kh=k/2")

            counting_results1, counting_exacts1 = read_sim_data(df, f"counting_bucket_{bucket_type}_{error_type}_k=k_kh=0")
            counting_results2, counting_exacts2 = read_sim_data(df, f"counting_bucket_{bucket_type}_{error_type}_k=k/2_kh=k/2")

            sampling_results1, sampling_exacts1 = read_sim_data(df, f"sampling_bucket_{bucket_type}_{error_type}_k=k/16_ku=16_kh=0")
            sampling_results2, sampling_exacts2 = read_sim_data(df, f"sampling_bucket_{bucket_type}_{error_type}_k=k/32_ku=16_kh=k/2")

            plot_mse(ax[i][j], sample_sizes, central_results1, central_exacts1, "Central estimator:\n$B = k, k_h = 0$")
            plot_mse(ax[i][j], sample_sizes, central_results2, central_exacts2, "Central estimator:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
            
            plot_mse(ax[i][j], sample_sizes, counting_results1, counting_exacts1, "Counting estimator:\n$B = k, k_h = 0$")
            plot_mse(ax[i][j], sample_sizes, counting_results2, counting_exacts2, "Counting estimator:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
            
            plot_mse(ax[i][j], sample_sizes, sampling_results1, sampling_exacts1, "Sampling estimator:\n$B = \\frac{k}{16}, k_u = 16, k_h = 0$")
            plot_mse(ax[i][j], sample_sizes, sampling_results2, sampling_exacts2, "Sampling estimator:\n$B = \\frac{k}{32}, k_u = 16, k_h = \\frac{k}{2}$")

            error_name = error_prefix(error_type)
            bucket_name = "Linear" if bucket_type == "linear" else "Exponential"
            ax[i][j].set_title(f'{bucket_name} bucket, {error_name} error')

    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')

    ax[1][0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1][2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Space size', y=-0.1)
    fig.supylabel('RMSPE (log-scale)')

    fig.suptitle(title)

    fig.legend(*ax[0][0].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/{file_name}', bbox_inches='tight')
    plt.show()

plot_bucket_schemes(deg3_aol, 2 ** np.arange(6, 17),
                 "3rd Frequency Moment RMSPE on AOL data across bucketing schemes",
                 "3rd_moment_bucket_schemes_aol")
plot_bucket_schemes(deg4_aol, 2 ** np.arange(6, 17),
                 "4th Frequency Moment RMSPE on AOL data across bucketing schemes",
                 "4th_moment_bucket_schemes_aol")

# plot_bucket_schemes(deg3_caida, 2 ** np.arange(6, 17),
#                  "3rd Frequency Moment RMSPE on CAIDA data across bucketing schemes",
#                  "3rd_moment_bucket_schemes_caida")
# plot_bucket_schemes(deg4_caida, 2 ** np.arange(6, 17),
#                  "4th Frequency Moment RMSPE on CAIDA data across bucketing schemes",
#                  "4th_moment_bucket_schemes_caida")
# plot_bucket_nrmse(threshold_fake_0_1, 'Synthetic $\\alpha=0.1$', 'fake_0_1')
# plot_bucket_nrmse(threshold_fake_0_3, 'Synthetic $\\alpha=0.3$', 'fake_0_3')
# plot_bucket_nrmse(threshold_fake_0_5, 'Synthetic $\\alpha=0.5$', 'fake_0_5')

# +
# Top k_h

def plot_bucket_top_kh(df, sample_sizes, title, file_name):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    error_types = ["rel_0.05", "abs_0.001", "train"]

    for i, error_type in enumerate(error_types):
        central_results1, central_exacts1 = read_sim_data(df, f"central_bucket_expo_{error_type}_k=k_kh=0")
        central_results2, central_exacts2 = read_sim_data(df, f"central_bucket_expo_{error_type}_k=k/2_kh=k/2")

        unbiased_results1, unbiased_exacts1 = read_sim_data(df, f"unbiased_bucket_{error_type}_kh=0")
        unbiased_results2, unbiased_exacts2 = read_sim_data(df, f"unbiased_bucket_{error_type}_kh=k")

        counting_results1, counting_exacts1 = read_sim_data(df, f"counting_bucket_expo_{error_type}_k=k_kh=0")
        counting_results2, counting_exacts2 = read_sim_data(df, f"counting_bucket_expo_{error_type}_k=k/2_kh=k/2")

        sampling_results1, sampling_exacts1 = read_sim_data(df, f"sampling_bucket_expo_{error_type}_k=k/16_ku=16_kh=0")
        sampling_results2, sampling_exacts2 = read_sim_data(df, f"sampling_bucket_expo_{error_type}_k=k/32_ku=16_kh=k/2")

        plot_mse(ax[i], sample_sizes, central_results1, central_exacts1, "Central estimator:\n$B = k, k_h = 0$")
        plot_mse(ax[i], sample_sizes, central_results2, central_exacts2, "Central estimator:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")

        plot_mse(ax[i], sample_sizes, unbiased_results1, unbiased_exacts1, "Unbiased estimator:\n$k_h = 0$")
        plot_mse(ax[i], sample_sizes, unbiased_results2, unbiased_exacts2, "Unbiased estimator:\n$k_h = k$")
        
        plot_mse(ax[i], sample_sizes, counting_results1, counting_exacts1, "Counting estimator:\n$B = k, k_h = 0$")
        plot_mse(ax[i], sample_sizes, counting_results2, counting_exacts2, "Counting estimator:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
        
        plot_mse(ax[i], sample_sizes, sampling_results1, sampling_exacts1, "Sampling estimator:\n$B = \\frac{k}{16}, k_u = 16, k_h = 0$")
        plot_mse(ax[i], sample_sizes, sampling_results2, sampling_exacts2, "Sampling estimator:\n$B = \\frac{k}{32}, k_u = 16, k_h = \\frac{k}{2}$")

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Space size', y=-0.1)
    fig.supylabel('RMSPE (log-scale)')

    fig.suptitle(title)
    ax[0].set_title("Relative error")
    ax[1].set_title("Absolute error")
    ax[2].set_title("Train/test error")

    fig.legend(*ax[0].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/{file_name}', bbox_inches='tight')
    plt.show()

plot_bucket_top_kh(deg3_aol, 2 ** np.arange(6, 17),
                 "3rd Frequency Moment RMSPE on AOL data with and without Top $k_h$ sample",
                 "3rd_moment_bucket_top_kh_aol")
plot_bucket_top_kh(deg4_aol, 2 ** np.arange(6, 17),
                 "4th Frequency Moment RMSPE on AOL data with and without Top $k_h$ sample",
                 "4th_moment_bucket_top_kh_aol")
# plot_bucket_top_kh(deg3_caida, 2 ** np.arange(6, 17),
#                  "3rd Frequency Moment RMSPE on CAIDA data with and without Top $k_h$ sample",
#                  "3rd_moment_bucket_top_kh_caida")
# plot_bucket_top_kh(deg4_caida, 2 ** np.arange(6, 17),
#                  "4th Frequency Moment RMSPE on CAIDA data with and without Top $k_h$ sample",
#                  "4th_moment_bucket_top_kh_caida")
# plot_bucket_nrmse(threshold_fake_0_1, 'Synthetic $\\alpha=0.1$', 'fake_0_1')
# plot_bucket_nrmse(threshold_fake_0_3, 'Synthetic $\\alpha=0.3$', 'fake_0_3')
# plot_bucket_nrmse(threshold_fake_0_5, 'Synthetic $\\alpha=0.5$', 'fake_0_5')

# +
# Top k_h

def plot_bucket_swa(df, sample_sizes, title, file_name):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    error_types = ["rel_0.05", "abs_0.001", "train"]

    for i, error_type in enumerate(error_types):
        central_results, central_exacts = read_sim_data(df, f"central_bucket_expo_{error_type}_k=k/2_kh=k/2")
        unbiased_results, unbiased_exacts = read_sim_data(df, f"unbiased_bucket_{error_type}_kh=k")
        counting_results, counting_exacts = read_sim_data(df, f"counting_bucket_expo_{error_type}_k=k/2_kh=k/2")
        sampling_results, sampling_exacts = read_sim_data(df, f"sampling_bucket_expo_{error_type}_k=k/32_ku=16_kh=k/2")

        swa_results1, swa_exacts3 = read_sim_data(df, f"swa_{error_type}_kh=0_kp=k_ku=0")
        swa_results2, swa_exacts2 = read_sim_data(df, f"swa_{error_type}_kh=k/2_kp=k/2_ku=0")
        swa_results3, swa_exacts1 = read_sim_data(df, f"swa_{error_type}_kh=k/2_kp=k/4_ku=k/4")

        ppswor_results, ppswor_exacts = read_sim_data(df, "ppswor")

        plot_mse(ax[i], sample_sizes, central_results, central_exacts, "Central estimator:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
        plot_mse(ax[i], sample_sizes, unbiased_results, unbiased_exacts, "Unbiased estimator:\n$k_h = k$")
        plot_mse(ax[i], sample_sizes, counting_results, counting_exacts, "Counting estimator:\n$B = \\frac{k}{2}, k_h = \\frac{k}{2}$")
        plot_mse(ax[i], sample_sizes, sampling_results, sampling_exacts, "Sampling estimator:\n$B = \\frac{k}{32}, k_u = 16, k_h = \\frac{k}{2}$")

        plot_mse(ax[i], sample_sizes, swa_results1, swa_exacts1, "SWA estimator:\n$k_h = 0, k_p = k, k_u = 0$")
        plot_mse(ax[i], sample_sizes, swa_results2, swa_exacts2, "SWA estimator:\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{2}, k_u = 0$")
        plot_mse(ax[i], sample_sizes, swa_results3, swa_exacts3, "SWA estimator:\n$k_h = \\frac{k}{2}, k_p = \\frac{k}{4}, k_u = \\frac{k}{4}$")

        plot_mse(ax[i], sample_sizes, ppswor_results, ppswor_exacts, "PPSWOR")

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[0].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[1].set_xticks(sample_sizes, sample_sizes, rotation='vertical')
    ax[2].set_xticks(sample_sizes, sample_sizes, rotation='vertical')

    fig.supxlabel('Space size', y=-0.1)
    fig.supylabel('RMSPE (log-scale)')

    fig.suptitle(title)
    ax[0].set_title("Relative error")
    ax[1].set_title("Absolute error")
    ax[2].set_title("Train/test error")

    fig.legend(*ax[0].get_legend_handles_labels(), bbox_to_anchor=(0.9, 0.5), loc='center left')

    plt.savefig(f'figs/{file_name}', bbox_inches='tight')
    plt.show()

plot_bucket_swa(deg3_aol, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 3rd Frequency Moment RMSPE on AOL data",
                 "3rd_moment_bucket_swa_aol")
plot_bucket_swa(deg4_aol, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 4th Frequency Moment RMSPE on AOL data",
                 "4th_moment_bucket_swa_aol")
plot_bucket_swa(deg3_0_1, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 3rd Frequency Moment RMSPE on $\\alpha=0.1$ Synthetic data",
                 "3rd_moment_bucket_swa_syn_0_1")
plot_bucket_swa(deg4_0_1, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 4th Frequency Moment RMSPE on $\\alpha=0.1$ Synthetic data",
                 "4th_moment_bucket_swa_syn_0_1")
plot_bucket_swa(deg3_0_3, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 3rd Frequency Moment RMSPE on $\\alpha=0.3$ Synthetic data",
                 "3rd_moment_bucket_swa_syn_0_3")
plot_bucket_swa(deg4_0_3, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 4th Frequency Moment RMSPE on $\\alpha=0.3$ Synthetic data",
                 "4th_moment_bucket_swa_syn_0_3")
plot_bucket_swa(deg3_0_5, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 3rd Frequency Moment RMSPE on $\\alpha=0.5$ Synthetic data",
                 "3rd_moment_bucket_swa_syn_0_5")
plot_bucket_swa(deg4_0_5, 2 ** np.arange(6, 17),
                 "SWA vs. Bucketing sketch, 4th Frequency Moment RMSPE on $\\alpha=0.5$ Synthetic data",
                 "4th_moment_bucket_swa_syn_0_5")
# plot_bucket_swa(deg3_caida, 2 ** np.arange(6, 17),
#                  "SWA vs. Bucketing sketch, 3rd Frequency Moment RMSPE on CAIDA data",
#                  "3rd_moment_bucket_swa_caida")
# plot_bucket_swa(deg4_caida, 2 ** np.arange(6, 17),
#                  "SWA vs. Bucketing sketch, 4th Frequency Moment RMSPE on CAIDA data",
#                  "4th_moment_bucket_swa_caida")
# plot_bucket_swa(threshold_aol, 2 ** np.arange(6, 13),
#                  "SWA vs. Bucketing sketch, Threshold RMSPE on AOL data",
#                  "threshold_bucket_swa_aol")
# plot_bucket_swa(threshold_caida, 2 ** np.arange(6, 13),
#                  "SWA vs. Bucketing sketch, Threshold RMSPE on CAIDA data",
#                  "threshold_bucket_swa_caida")
# plot_bucket_swa(threshold_0_1, 2 ** np.arange(6, 13),
#                  "SWA vs. Bucketing sketch, Threshold RMSPE on $\\alpha=0.1$ Synthetic data",
#                  "threshold_bucket_swa_0_1")
# plot_bucket_swa(threshold_0_3, 2 ** np.arange(6, 13),
#                  "SWA vs. Bucketing sketch, Threshold RMSPE on $\\alpha=0.3$ Synthetic data",
#                  "threshold_bucket_swa_0_3")
# plot_bucket_swa(threshold_0_5, 2 ** np.arange(6, 13),
#                  "SWA vs. Bucketing sketch, Threshold RMSPE on $\\alpha=0.5$ Synthetic data",
#                  "threshold_bucket_swa_0_5")
