#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <random>

using namespace std;

struct MockOracle {
    MockOracle(double ep, unordered_map<string, size_t>& item_counts): ep(ep), item_counts(item_counts) {
        estimates.reserve(item_counts.size());
        reset_estimates();
    }

    inline double estimate(const string& item) { return estimates[item]; }

    inline void reset_estimates() {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(1-ep, 1+ep);

        for(auto& item : item_counts) {
            estimates[item.first] = d(gen) * item.second;
        }
    }

    double ep;
    unordered_map<string, double> estimates;

    unordered_map<string, size_t>& item_counts;
};

#endif