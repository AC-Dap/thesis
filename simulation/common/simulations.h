#ifndef SIMULATIONS_H
#define SIMULATIONS_H

#include <fstream>
#include <iostream>
#include <format>
#include <string>
#include <functional>
#include <csignal>
#include <vector>

#include "common/io/results.h"

using namespace std;

// Write mode to use for data file, if the file already exists.
enum FileWriteMode {
    SKIP,
    OVERWRITE,
};

// Flag to save results to file on abort
static volatile bool save_file = false;

inline void signal_handler(const int sig_code) {
    if (sig_code == SIGINT) {
        save_file = true;
    }
}

inline void run_sims(Results &results, const vector<size_t> &ks, const size_t n_sims,
                     const string& sketch_type, const function<vector<double>(size_t, size_t)> run_n_sims,
                     const __uint128_t exact_value, const FileWriteMode mode) {
    for (size_t k: ks) {
        // Get which trials we need to run
        vector<size_t> trials;
        for (int i = 1; i <= n_sims; i++) {
            if (mode == OVERWRITE || !results.has(sketch_type, k, i)) {
                trials.push_back(i);
            }
        }

        if (trials.empty()) {
            cout << results.output_path << " | Skipping " << sketch_type << " k=" << k << endl;
            continue;
        }

        // Run the trials
        cout << results.output_path << " | " << sketch_type << " k=" << k << ": ";
        vector<double> estimates = run_n_sims(k, trials.size());

        // Write trials to results
        for (size_t i = 0; i < trials.size(); i++) {
            results.add_result(sketch_type, k, trials[i], estimates[i], exact_value);
        }

        // Make sure our results file is updated with the newest sims
        if (save_file) {
            results.flush_to_file();
            exit(1);
        }
    }
}

#endif //SIMULATIONS_H
