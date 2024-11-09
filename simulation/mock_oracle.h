#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <concepts>

#include "dataset.h"

using namespace std;

struct MockOracle {
    MockOracle(double ep, Dataset& ds): ep(ep), ds(ds) {
        estimates.reserve(ds.item_counts.size());
    }

    inline double estimate(const string* item) { return estimates[item]; }

    virtual void reset_estimates(){};

    double ep;
    unordered_map<const string*, double> estimates;

    Dataset& ds;
};

struct MockOracleRelativeError : MockOracle {
    MockOracleRelativeError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(1-ep, 1+ep);

        for(auto& item : ds.item_counts) {
            estimates[item.first] = d(gen) * item.second;
        }
    }
};

struct MockOracleAbsoluteError : MockOracle {
    MockOracleAbsoluteError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(0, ep);

        double N = 0;
        for(auto& item : ds.item_counts) N += item.second;

        for(auto& item : ds.item_counts) {
            estimates[item.first] = item.second + d(gen) * N;
        }
    }
};

template <class T>
concept IsMockOracle = std::is_base_of<MockOracle, T>::value;

#endif