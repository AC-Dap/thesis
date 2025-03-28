import numpy as np
import pandas as pd
import os
import sys
import subprocess


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
    subprocess.run(["python3", "../data/generate_fake_dataset.py", str(a)], cwd="../data")

    print("Processing datasets...")
    subprocess.run(["python3", "../data/process_fake_dataset.py", str(a)], cwd="../data")

    # Run ./out/sim on each dataset
    print("Running simulations...")
    for i in range(N_DATASETS, 0, -1):
        output_name = f"fake_{a}"
        processed_data_path = f"../data/processed/fake_{a}_dataset"
        subprocess.run(["./out/sim", "1",
                        f"{processed_data_path}/train.txt",
                        f"{processed_data_path}/test_{i}.txt",
                        output_name])

        # Renumber latest simulation's trial number so it doesn't get overwritten
        def update_trial_numbers(csv_name):
            df = pd.read_csv(csv_name)

            # Get existing combinations of sketch_type, k, and the new n_trial value
            existing_combinations = df[df["n_trial"] == i][["sketch_type", "k"]]

            # Create mask checking both columns aren't already present with new n_trial
            mask = (df["n_trial"] == 1) & ~df[["sketch_type", "k"]].apply(tuple, axis=1).isin(
                existing_combinations.apply(tuple, axis=1)
            )
            df.loc[mask, "n_trial"] = i
            df.to_csv(csv_name, index=False)

        update_trial_numbers(f"results/{output_name}_threshold.csv")
        update_trial_numbers(f"results/{output_name}_moments_deg=3.csv")
        update_trial_numbers(f"results/{output_name}_moments_deg=4.csv")
