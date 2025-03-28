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
import sys


# %%
def process_fake_dataset(folder):
    files = ['train.txt'] + [f'test_{i}.txt' for i in range(1, 11)]
    mapping = {}
    nextId = 0
    with open(f"processed/{folder}/mapping.txt", "w") as f_map:
        for file in files:
            with open(f'{folder}/{file}', 'r') as src, open(f'processed/{folder}/{file}', 'w') as dst:
                for line in src:
                    el = float(line.strip())
                    if el not in mapping:
                        mapping[el] = nextId
                        f_map.write(f"{el}\n")
                        nextId += 1
                    dst.write(f'{mapping[el]}\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected usage: process_fake_dataset.py shape_parameter")
        print("Got:", sys.argv)
        exit(1)

    a = float(sys.argv[1])
    process_fake_dataset(f'fake_{a}_dataset')
