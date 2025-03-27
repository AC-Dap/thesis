#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <csignal>

#include "common/simulations.h"
#include "common/mock_oracle.h"
#include "common/io/dataset.h"
#include "threshold/simulations.h"
#include "moments/simulations.h"

using namespace std;

int main(int argc, const char** argv) {
    // Command line usage:
    // ./sim num_trials train_path test_path output_csv_name
    if (argc != 5) {
        throw std::invalid_argument("Wrong number of arguments: expected 4, got " + to_string(argc-1));
    }
    size_t total_trials = stoi(argv[1]);
    string train_path = argv[2];
    string test_path = argv[3];
    string output_name = argv[4];

    // Read in datasets
    size_t n_unique_items = count_unique_items({train_path, test_path});
    Dataset ds_train(n_unique_items), ds_test(n_unique_items);
    ds_train.add_from_file(train_path);
    ds_test.add_from_file(test_path);

    // Create oracles
    MockOracleAbsoluteError o_abs(0.001, "abs_0.001", ds_test);
    MockOracleRelativeError o_rel(0.05, "rel_0.05", ds_test);
    ExactOracle o_train("train", ds_train);
    vector<MockOracle*> os = {&o_abs, &o_rel, &o_train};

    // Set up signal handler
    signal(SIGINT, signal_handler);

    // Run all sims
    FileWriteMode mode = SKIP;
    // threshold::run_all_sims(ds_test, os, total_trials, output_name, mode);
    moments::run_all_sims(ds_test, os, total_trials, output_name, mode);
}
