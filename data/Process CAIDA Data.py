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
import os
import pandas as pd
import numpy as np
from dpkt.utils import inet_to_str

# %%
# Aggregate CSVs
folder = 'processed/CAIDA'
dfs = []
for filename in os.listdir(folder):
    if filename.startswith('orig-'):
        dfs.append(pd.read_csv(f"{folder}/{filename}"))
df = pd.concat(dfs)


# %%
# Create (src, dest) pairs
def process_row(src, dest):
    src_ip = inet_to_str(int(src).to_bytes(4))
    dest_ip = inet_to_str(int(dest).to_bytes(4))
    ip_pair = f"{src_ip},{dest_ip}"
    return ip_pair
df['src_dest_ips'] = df.apply(lambda row: process_row(row['src'], row['dest']), axis=1)

# %%
value_counts = df['src_dest_ips'].value_counts()

# %%
# Create mapping of entry -> id
mapping = {}
with open(f"{folder}/mapping.txt", "w") as f:
    for i, (key, _) in enumerate(value_counts.items()):
        mapping[key] = i
        f.write(f"{key}\n")

# %%
# Recreate CSVs using mapping instead
output_files = ["train.txt"] + [f"{str(i)}.txt" for i in range(1, 12)]
for i, output_file in enumerate(output_files):
    with open(f"{folder}/{output_file}", 'w') as f:
        start = i * 1_000_000
        end = (i+1) * 1_000_000
        for _, entry in df['src_dest_ips'].iloc[start:end].items():
            f.write(f"{mapping[entry]}\n")    
