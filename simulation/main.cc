#include <fstream>
#include <iostream>
#include <format>
#include <string>

#include "dataset.h"
#include "simulations.h"

using namespace std;

constexpr char const* DATA_PATH = "../data/AOL-user-ct-collection/user-ct-test-collection-01.txt";

void ppswor_sim(size_t k, size_t deg, size_t nsims, Dataset& ds) {
    auto ppswor_estimates = run_n_ppswor_sims(k, deg, nsims, ds);
    ofstream ppswor_out(format("results/ppswor_k={}_deg={}.txt", k, deg), ofstream::app);
    for(auto estimate : ppswor_estimates) ppswor_out << format("{}", estimate) << endl;
    ppswor_out.close();
}

void swa_sim(size_t kh, size_t kp, size_t ku, double ep, size_t deg, size_t nsims, Dataset& ds) {
    auto swa_estimates = run_n_swa_sims(kh, kp, ku, ep, deg, nsims, ds);
    ofstream swa_out(format("results/swa_k={}-{}-{}_ep={}_deg={}.txt", kh, kp, ku, ep, deg), ofstream::app);
    for(auto estimate : swa_estimates) swa_out << format("{}", estimate) << endl;
    swa_out.close();
}

int main() {
    Dataset ds;
    if(!read_dataset(DATA_PATH, ds)) {
        return -1;
    }

    const size_t deg = 3, nsims = 10;
    vector<size_t> ks = {1<<8, 1<<10, 1<<12};
//    vector<tuple<size_t, size_t, size_t>> swa_ks = {{0, 1<<8, 0}, {0, 1<<10, 0}, {0, 1<<12, 0},
//                                                     {0, 1<<7, 1<<7}, {0, 1<<9, 1<<9}, {0, 1<<11, 1<<11}};
    vector<tuple<size_t, size_t, size_t>> swa_ks = {{10, 1<<8, 0}, {10, 1<<10, 0}, {10, 1<<12, 0},
                                                     {10, 1<<7, 1<<7}, {10, 1<<9, 1<<9}, {10, 1<<11, 1<<11}};
    const double ep = 0.05;

//    for(auto k : ks) {
//        ppswor_sim(k, deg, nsims, ds);
//    }

    for(auto k : swa_ks) {
        swa_sim(get<0>(k), get<1>(k), get<2>(k), ep, deg, nsims, ds);
    }
}