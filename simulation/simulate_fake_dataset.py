import numpy as np
import pandas as pd
import os
import sys
import subprocess


N_ELEMENTS = 1000000
N_DATASETS = 30 # + 1 for training
rng = np.random.default_rng()

def generate_dataset(a):
    return np.floor(100 * (1 + np.random.pareto(a, size=N_ELEMENTS)))

def write_dataset(a, path):
    if os.path.exists(path):
        return

    data = generate_dataset(a)
    with open(path, 'w') as f:
        # Header line
        f.write("Index\tQuery\n")
        for i, el in enumerate(data):
            f.write(f"{i}\t{el}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected usage: simulate_fake_dataset.py shape_parameter")
        print("Got:", sys.argv)
        exit(1)

    a = float(sys.argv[1])
    directory_path = f"../data/fake_{a}_dataset"
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    print("Creating datasets...")
    write_dataset(a, f"{directory_path}/train.txt")
    for i in range(N_DATASETS):
        write_dataset(a, f"{directory_path}/test_{i + 1}.txt")

    # Run ./out/sim on each dataset
    print("Running simulations...")
    for i in range(N_DATASETS, 0, -1):
        subprocess.run(["./out/sim", "1",
                        f"{directory_path}/train.txt",
                        f"{directory_path}/test_{i}.txt",
                        f"fake_{a}_results"])

        # Renumber latest simulation's trial number so it doesn't get overwritten
        df3 = pd.read_csv(f"results/deg=3_fake_{a}_results.csv")
        df3.loc[df3["n_trial"] == 1, "n_trial"] = i
        df3.to_csv(f"results/deg=3_fake_{a}_results.csv", index=False)

        df4 = pd.read_csv(f"results/deg=4_fake_{a}_results.csv")
        df4.loc[df4["n_trial"] == 1, "n_trial"] = i
        df4.to_csv(f"results/deg=4_fake_{a}_results.csv", index=False)
