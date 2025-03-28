# Harvard Senior Thesis

This code repository contains all the code relevant to my senior thesis.

### Data

The `data` folder contains the various datasets for my experiments. This includes AOL search query data, CAIDA network traffic data, and synthetic data.

The AOL and CAIDA datasets should be downloaded; their sources are given in their folders. After they are donwloaded, they should be processed by running `process_aol_dataset.py` and `process_caida_dataset.py` respectively.

The synthetic data will be generated and processed automatically when running the experiments.

### Experiments

All the code to run experiments lives in `simulation`. They can be run with `make run_all`; this may take a while!

### Plot Generation

Code to generate all the figures lives in this directory, and places it in the `figs` folder.
- `Final Plot Generation.py` generates all the figures using simulation data.
- `Quantile Sketch.py` runs the quantile sketch and generates the relevant plots.
- `Oracle Error Plots.py` visualizes the oracle errors.
