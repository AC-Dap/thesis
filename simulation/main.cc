#include <fstream>
#include <iostream>
#include <chrono>
#include <format>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "ppswor.h"
#include "mock_oracle.h"
#include "swa.h"
#include "estimator.h"
#include "fake_swa.h"
#include "dataset.h"
#include "simulations.h"

using namespace std;

constexpr char const* DATA_PATH = "../data/AOL-user-ct-collection/user-ct-test-collection-01.txt";

int main() {
    Dataset ds;
    if(!read_dataset(DATA_PATH, ds)) {
        return -1;
    }

    const size_t deg = 3, nsims = 1;
    const size_t k = 1<<10, kh = 0, kp = 1<<10, ku = 0;
    const double ep = 0.05;

    auto ppswor_estimates = run_n_ppswor_sims(k, deg, nsims, ds);
    ofstream ppswor_out(format("ppswor_k={}_deg={}_nsims={}.txt", k, deg, nsims));
    ppswor_out << "k deg nsims" << endl;
    ppswor_out << format("{} {} {}", k, deg, nsims) << endl;
    for(auto estimate : ppswor_estimates) ppswor_out << format("{}", estimate) << endl;
    ppswor_out.close();

//    auto swa_estimates = run_n_swa_sims(kh, kp, ku, ep, deg, nsims, ds);
//    ofstream swa_out(format("swa_k={}-{}-{}_ep={}_deg={}_nsims={}.txt", kh, kp, ku, ep, deg, nsims));
//    swa_out << "kh kp ku ep deg nsims" << endl;
//    swa_out << format("{} {} {} {} {} {}", kh, kp, ku, ep, deg, nsims) << endl;
//    for(auto estimate : swa_estimates) swa_out << format("{}", estimate) << endl;
//    swa_out.close();
}