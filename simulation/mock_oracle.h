#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <random>

#include "dataset.h"

using namespace std;

struct MockOracle {
    MockOracle(double ep, Dataset& ds): ep(ep), ds(ds) {
        estimates.reserve(ds.item_counts.size());
        reset_estimates();
    }

    inline double estimate(const string* item) { return estimates[item]; }

    inline void reset_estimates() {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(1-ep, 1+ep);

        for(auto& item : ds.item_counts) {
            estimates[item.first] = d(gen) * item.second;
        }
    }

    double ep;
    unordered_map<const string*, double> estimates;

    Dataset& ds;
};

#endif