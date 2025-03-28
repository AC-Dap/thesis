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
import pandas as pd
import numpy as np

# %%
train_file = "AOL-user-ct-collection/user-ct-test-collection-01.txt"
test_file = "AOL-user-ct-collection/user-ct-test-collection-02.txt"

dfs = []
for file in [train_file, test_file]:
    df = pd.read_csv(file, sep='\t')
    dfs.append(df)
df = pd.concat(dfs)
unique_queries = df['Query'].unique()

# %%
len(unique_queries)

# %%
mapping = {}
folder = 'processed/AOL'
with open(f"{folder}/mapping.txt", "w") as f:
    for i, key in enumerate(unique_queries):
        mapping[key] = i
        f.write(f"{key}\n")


# %%
def process_df(df, mapping, output_file):
    with open(f"{folder}/{output_file}", "w") as f:
        for _, query in df['Query'].items():
            f.write(f"{mapping[query]}\n")

process_df(dfs[0], mapping, 'train.txt')
process_df(dfs[1], mapping, 'test.txt')
