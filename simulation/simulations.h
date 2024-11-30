#ifndef SIMULATIONS_H
#define SIMULATIONS_H

#include <unordered_map>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <concepts>

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

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Initializing PPSWOR sketch took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

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

    auto t3 = chrono::high_resolution_clock::now();
    cout << "Running simulations took "
         << chrono::duration_cast<chrono::milliseconds>(t3-t2).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

template <class T>
requires IsMockOracle<T>
vector<double> run_n_swa_sims(size_t kh, size_t kp, size_t ku, double oracle_ep, size_t deg, size_t nsims, Dataset& ds){
    auto t1 = chrono::high_resolution_clock::now();

    T o(oracle_ep, ds);

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Initializing oracle took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

    vector<double> estimates(nsims);
    for(int i = 0; i < nsims; i++) {
        o.reset_estimates();

        auto sample = fake_swa_sample(kh, kp, ku, deg, o, ds);
        auto weights = get<1>(sample), probs = get<2>(sample);

        estimates[i] = estimate_moment(weights, probs, deg);

        cout << "." << flush;
    }
    cout << endl;

    auto t3 = chrono::high_resolution_clock::now();
    cout << "Running simulations took "
         << chrono::duration_cast<chrono::milliseconds>(t3-t2).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

template <class T>
requires IsMockOracle<T>
vector<double> run_n_expo_bucket_sims(double min_freq, size_t k, size_t k_hh, const function<double(double, double, double)>& n_estimate, double oracle_ep, size_t deg, size_t nsims, Dataset& ds){
    auto t1 = chrono::high_resolution_clock::now();

    Buckets b = generate_exponential_buckets(min_freq, k);
    T o(oracle_ep, ds);

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Initializing oracle took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

    vector<double> estimates(nsims);
    for(int i = 0; i < nsims; i++) {
        o.reset_estimates();

        estimates[i] = bucket_sketch(k_hh, deg, n_estimate, b, o, ds);

        cout << "." << flush;
    }
    cout << endl;

    auto t3 = chrono::high_resolution_clock::now();
    cout << "Running simulations took "
         << chrono::duration_cast<chrono::milliseconds>(t3-t2).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

template <class T>
requires IsMockOracle<T>
vector<double> run_n_linear_bucket_sims(size_t k, size_t k_hh, const function<double(double, double, double)>& n_estimate, double oracle_ep, size_t deg, size_t nsims, Dataset& ds){
    auto t1 = chrono::high_resolution_clock::now();

    Buckets b = generate_linear_buckets(k);
    T o(oracle_ep, ds);

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Initializing oracle took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

    vector<double> estimates(nsims);
    for(int i = 0; i < nsims; i++) {
        o.reset_estimates();

        estimates[i] = bucket_sketch(k_hh, deg, n_estimate, b, o, ds);

        cout << "." << flush;
    }
    cout << endl;

    auto t3 = chrono::high_resolution_clock::now();
    cout << "Running simulations took "
         << chrono::duration_cast<chrono::milliseconds>(t3-t2).count() / 1000.0 / nsims
         << " seconds on average" << endl;

    return estimates;
}

#endif