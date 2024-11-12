#include <fstream>
#include <iostream>
#include <format>
#include <string>
#include <concepts>

#include "dataset.h"
#include "estimator.h"
#include "simulations.h"

using namespace std;

constexpr char const* DATA_PATH = "../data/AOL-user-ct-collection/user-ct-test-collection-01.txt";

void ppswor_sim(size_t k, size_t deg, size_t nsims, Dataset& ds) {
    auto ppswor_estimates = run_n_ppswor_sims(k, deg, nsims, ds);
    ofstream ppswor_out(format("results/ppswor_k={}_deg={}.txt", k, deg), ofstream::app);
    for(auto estimate : ppswor_estimates) ppswor_out << format("{}", estimate) << endl;
    ppswor_out.close();
}

template <class T>
requires IsMockOracle<T>
void swa_sim(size_t kh, size_t kp, size_t ku, double ep, size_t deg, size_t nsims, Dataset& ds) {
    auto swa_estimates = run_n_swa_sims<T>(kh, kp, ku, ep, deg, nsims, ds);
    ofstream swa_out(format("results/swa_k={}-{}-{}_abs_ep={}_deg={}.txt", kh, kp, ku, ep, deg), ofstream::app);
    for(auto estimate : swa_estimates) swa_out << format("{}", estimate) << endl;
    swa_out.close();
}

void exact_sim(size_t kp, size_t deg, size_t nsims, Dataset& ds) {
    auto swa_estimates = run_n_swa_sims<ExactOracle>(0, kp, 0, 0, deg, nsims, ds);
    ofstream swa_out(format("results/exact_k={}_deg={}.txt", kp, deg), ofstream::app);
    for(auto estimate : swa_estimates) swa_out << format("{}", estimate) << endl;
    swa_out.close();
}

int main() {
    Dataset ds;
    if(!ds.read_from_file(DATA_PATH)) {
        return -1;
    }

    const size_t deg = 4, nsims = 30;
    vector<size_t> ks = {1<<9, 1<<11, 1<<13, 1<<15};
    vector<tuple<size_t, size_t, size_t>> swa_ks;
    for(auto k : ks) {
        swa_ks.push_back({0, k, 0});
        swa_ks.push_back({100, k, 0});
        swa_ks.push_back({0, k/2, k/2});
        swa_ks.push_back({100, k/2, k/2});
    }

    const double ep = 0.05;

    cout << "Exact moment: " << format("{}", exact_moment(ds.item_counts, deg)) << endl;

//    for(auto k : ks) {
//        ppswor_sim(k, 3, nsims, ds);
//        ppswor_sim(k, 4, nsims, ds);
//    }
//
//    for(auto k : swa_ks) {
//        swa_sim<MockOracleAbsoluteError>(get<0>(k), get<1>(k), get<2>(k), ep, 3, nsims, ds);
//        swa_sim<MockOracleAbsoluteError>(get<0>(k), get<1>(k), get<2>(k), ep, 4, nsims, ds);
//        swa_sim<MockOracleRelativeError>(get<0>(k), get<1>(k), get<2>(k), ep, 3, nsims, ds);
//        swa_sim<MockOracleRelativeError>(get<0>(k), get<1>(k), get<2>(k), ep, 4, nsims, ds);
//    }

//    for(auto k : ks) {
//        exact_sim(k, 3, nsims, ds);
//        exact_sim(k, 4, nsims, ds);
//    }
}