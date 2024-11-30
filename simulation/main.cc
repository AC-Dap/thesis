#include <fstream>
#include <iostream>
#include <format>
#include <string>
#include <concepts>
#include <functional>

#include "dataset.h"
#include "estimator.h"
#include "simulations.h"

using namespace std;

constexpr char const* DATA_PATH = "../data/AOL-user-ct-collection/user-ct-test-collection-01.txt";

// Write mode to use for data file, if the file already exists.
enum FileWriteMode {
    SKIP,
    TRUNC,
    APPEND
};

void write_to_file(const string& file_name, const function<vector<double>()>& sim_function, FileWriteMode mode) {
    if(mode == SKIP) {
        if(ifstream in(file_name); in.good()) {
            cout << "Skipping " << file_name << endl;
            return;
        }
    }

    ofstream out(file_name, mode == TRUNC ? ofstream::trunc : ofstream::app);
    cout << "Writing to " << file_name << endl;

    auto estimates = sim_function();
    for(auto estimate : estimates)
        out << format("{:a}", estimate) << endl;
    out.close();
}

void ppswor_sim(size_t k, size_t deg, size_t nsims, Dataset& ds, FileWriteMode mode = SKIP) {
    string file_name = format("results/deg={}/ppswor/ppswor_k={}_deg={}.txt", deg, k, deg);
    write_to_file(file_name, [&] {
        return run_n_ppswor_sims(k, deg, nsims, ds);
    }, mode);
}

void exact_sim(size_t kp, size_t deg, size_t nsims, Dataset& ds, FileWriteMode mode = SKIP) {
    string file_name = format("results/deg={}/exact/exact_k={}_deg={}.txt", deg, kp, deg);
    write_to_file(file_name, [&] {
        return run_n_swa_sims<ExactOracle>(0, kp, 0, 0, deg, nsims, ds);
    }, mode);
}

template <class T>
requires IsMockOracle<T>
void swa_sim(size_t kh, size_t kp, size_t ku, double ep, size_t deg, size_t nsims, Dataset& ds, FileWriteMode mode = SKIP) {
    string file_name = format("results/deg={}/swa_{}/swa_{}_k={}-{}-{}_ep={}_deg={}.txt", deg, T::prefix, T::prefix, kh, kp, ku, ep, deg);
    write_to_file(file_name, [&] {
        return run_n_swa_sims<T>(kh, kp, ku, ep, deg, nsims, ds);
    }, mode);
}

template <class T>
requires IsMockOracle<T>
void expo_bucket_sim(double min_freq, size_t k, size_t k_hh, const function<double(double, double, double)>& n_estimate, string n_est_type, double ep, size_t deg, size_t nsims, Dataset& ds, FileWriteMode mode = SKIP) {
    string file_name = format("results/deg={}/expo_bucket_{}_{}/expo_bucket_{}_{}_k={}_khh={}_min_freq={}_ep={}_deg={}.txt",
        deg, n_est_type, T::prefix, n_est_type, T::prefix, k, k_hh, min_freq, ep, deg);
    write_to_file(file_name, [&] {
        return run_n_expo_bucket_sims<T>(min_freq, k, k_hh, n_estimate, ep, deg, nsims, ds);
    }, mode);
}

template <class T>
requires IsMockOracle<T>
void linear_bucket_sim(size_t k, size_t k_hh, const function<double(double, double, double)>& n_estimate, string n_est_type, double ep, size_t deg, size_t nsims, Dataset& ds, FileWriteMode mode = SKIP) {
    string file_name = format("results/deg={}/linear_bucket_{}_{}/linear_bucket_{}_{}_k={}_khh={}_ep={}_deg={}.txt",
        deg, n_est_type, T::prefix, n_est_type, T::prefix, k, k_hh, ep, deg);
    write_to_file(file_name, [&] {
        return run_n_linear_bucket_sims<T>(k, k_hh, n_estimate, ep, deg, nsims, ds);
    }, mode);
}

int main() {
    Dataset ds;
    if(!ds.read_from_file(DATA_PATH)) {
        return -1;
    }

    constexpr size_t nsims = 30;
    constexpr double ep = 0.05;
    constexpr size_t min_freq = 7;

    vector<size_t> degs = {3, 4};
    vector<size_t> ks = {1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15, 1<<16};

    FileWriteMode mode = SKIP;
    for (auto deg : degs) {
        for (auto k : ks) {
            // PPSWOR
            ppswor_sim(k, deg, nsims, ds, mode);

            // Exact PPSWOR
            exact_sim(k, deg, nsims, ds, mode);

            // SWA
            vector<tuple<size_t, size_t, size_t>> swa_ks = {{0, k , 0}, {k/2, k/2, 0}, {0, k/2, k/2}};
            for(auto [k_h, k_p, k_u] : swa_ks) {
                swa_sim<MockOracleAbsoluteError>(k_h, k_p, k_u, ep, deg, nsims, ds, mode);
                swa_sim<MockOracleBinomialError>(k_h, k_p, k_u, ep, deg, nsims, ds, mode);
                swa_sim<MockOracleRelativeError>(k_h, k_p, k_u, ep, deg, nsims, ds, mode);
            }

            // Exponential bucket
            vector<tuple<size_t, size_t>> bucket_ks = {{k, 0}, {k/2, k/2}};
            vector<tuple<function<double(double, double, double)>, string>> n_estimates = {
                {n_estimate_left, "lower"},
                    {n_estimate_right, "upper"},
                {n_estimate_arith_avg, "arith"},
                {n_estimate_geo_avg, "geo"},
                {n_estimate_harm_avg, "harm"}
            };
            for (auto [k, k_hh] : bucket_ks) {
                for(auto& [n_estimate, name] : n_estimates) {
                    expo_bucket_sim<MockOracleAbsoluteError>(min_freq, k, k_hh, n_estimate, name, ep, deg, nsims, ds, mode);
                    expo_bucket_sim<MockOracleBinomialError>(min_freq, k, k_hh, n_estimate, name, ep, deg, nsims, ds, mode);
                    expo_bucket_sim<MockOracleRelativeError>(min_freq, k, k_hh, n_estimate, name, ep, deg, nsims, ds, mode);
                    expo_bucket_sim<ExactOracle>(min_freq, k, k_hh, n_estimate, name, ep, deg, 1, ds, mode);

                    linear_bucket_sim<MockOracleAbsoluteError>(k, k_hh, n_estimate, name, ep, deg, nsims, ds, mode);
                    linear_bucket_sim<MockOracleBinomialError>(k, k_hh, n_estimate, name, ep, deg, nsims, ds, mode);
                    linear_bucket_sim<MockOracleRelativeError>(k, k_hh, n_estimate, name, ep, deg, nsims, ds, mode);
                    linear_bucket_sim<ExactOracle>(k, k_hh, n_estimate, name, ep, deg, 1, ds, mode);
                }
            }
        }
    }
}