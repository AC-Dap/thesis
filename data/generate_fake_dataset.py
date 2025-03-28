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
import os
import sys

# %%
N_ELEMENTS = 1000000
N_DATASETS = 10 # + 1 for training
rng = np.random.default_rng()

def generate_dataset(a):
    return np.floor(100 * (1 + np.random.pareto(a, size=N_ELEMENTS)))

def write_dataset(a, path):
    if os.path.exists(path):
        return

    data = generate_dataset(a)
    with open(path, 'w') as f:
        for i, el in enumerate(data):
            f.write(f"{el}\n")

# %%
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected usage: generate_fake_dataset.py shape_parameter")
        print("Got:", sys.argv)
        exit(1)
    
    a = float(sys.argv[1])
    directory_path = f"fake_{a}_dataset"
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    
    print("Creating datasets...")
    write_dataset(a, f"{directory_path}/train.txt")
    for i in range(N_DATASETS):
        write_dataset(a, f"{directory_path}/test_{i + 1}.txt")
