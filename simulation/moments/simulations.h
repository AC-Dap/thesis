#ifndef MOMENTS_SIMULATIONS_H
#define MOMENTS_SIMULATIONS_H

#include <vector>
#include <string>

#include "common/mock_oracle.h"
#include "common/simulations.h"
#include "common/bucket_sketch.h"
#include "common/io/dataset.h"

using namespace std;

namespace moments {

    void run_all_sims(Dataset& ds, vector<MockOracle*>& oracles,
        size_t total_trials, string& output_name, FileWriteMode mode);

    vector<double> run_n_ppswor_sims(size_t k, size_t deg, size_t nsims, Dataset& ds);

    vector<double> run_n_swa_sims(size_t kh, size_t kp, size_t ku, MockOracle& o, size_t deg, size_t nsims, Dataset& ds);

    vector<double> run_n_bucket_sims(
        function<Buckets()> bucket_gen,
        function<double(Buckets&, MockOracle&, Dataset&)> sketch,
        size_t nsims, MockOracle& o, Dataset& ds);
}

#endif