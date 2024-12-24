#ifndef SIMULATIONS_H
#define SIMULATIONS_H

#include <vector>
#include <string>
#include <chrono>
#include <iostream>

#include "ppswor.h"
#include "estimator.h"
#include "fake_swa.h"
#include "mock_oracle.h"
#include "dataset.h"
#include "bucket_sketch.h"

using namespace std;

vector<double> run_n_ppswor_sims(size_t k, size_t deg, size_t nsims, Dataset& ds){
    auto t1 = chrono::high_resolution_clock::now();

    PPSWOR pp(k, 2, ds);

    vector<double> estimates(nsims);
    for(int i = 0; i < nsims; i++) {
        if(i != 0) pp.reset();

        for(auto item : ds.lines) {
            pp.update(item, 1);
        }

        auto sample = pp.sample();
        auto weights = get<1>(sample), probs = get<2>(sample);

        estimates[i] = estimate_moment(weights, probs, deg);

        cout << "." << flush;
    }
    cout << endl;

    auto t2 = chrono::high_resolution_clock::now();
    cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

vector<double> run_n_swa_sims(size_t kh, size_t kp, size_t ku, MockOracle& o, size_t deg, size_t nsims, Dataset& ds){
    auto t1 = chrono::high_resolution_clock::now();

    vector<double> estimates(nsims);
    for(int i = 0; i < nsims; i++) {
        o.reset_estimates();

        auto sample = fake_swa_sample(kh, kp, ku, deg, o, ds);
        auto weights = get<1>(sample), probs = get<2>(sample);

        estimates[i] = estimate_moment(weights, probs, deg);

        cout << "." << flush;
    }
    cout << endl;

    auto t2 = chrono::high_resolution_clock::now();
    cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

vector<double> run_n_bucket_sims(
    const function<Buckets()> bucket_gen,
    const function<double(Buckets&, MockOracle&, Dataset&)> sketch,
    size_t nsims, MockOracle& o, Dataset& ds) {
    auto t1 = chrono::high_resolution_clock::now();

    Buckets b = bucket_gen();

    vector<double> estimates(nsims);
    for(int i = 0; i < nsims; i++) {
        o.reset_estimates();

        estimates[i] = sketch(b, o, ds);

        cout << "." << flush;
    }
    cout << endl;

    auto t2 = chrono::high_resolution_clock::now();
    cout << " | " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

#endif