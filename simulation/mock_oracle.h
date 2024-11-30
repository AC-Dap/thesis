#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <unordered_map>
#include <random>

#include "dataset.h"

using namespace std;

struct MockOracle {
    MockOracle(double ep, Dataset& ds): ep(ep), ds(ds) {
        estimates.reserve(ds.item_counts.size());
    }

    double estimate(const string* item) { return estimates[item]; }

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

        uniform_real_distribution<> d(1.-ep, 1.+ep);

        for(auto& item : ds.item_counts) {
            estimates[item.first] = d(gen) * item.second;
        }
    }

    static constexpr const char* prefix = "rel";
};

struct MockOracleAbsoluteError : MockOracle {
    MockOracleAbsoluteError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(-ep, ep);

        for(auto& item : ds.item_counts) {
            estimates[item.first] = item.second + d(gen) * ds.lines.size();
        }
    }

    static constexpr const char* prefix = "abs";
};

struct MockOracleBinomialError : MockOracle {
    MockOracleBinomialError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        double N = ds.lines.size();

        for(auto& item : ds.item_counts) {
            binomial_distribution<> d(N, item.second/N);

            estimates[item.first] = d(gen);
        }
    }

    static constexpr const char* prefix = "bin";
};

struct ExactOracle : MockOracle {
    ExactOracle(double ep, Dataset& ds): MockOracle(ep, ds){}

    void reset_estimates() override {
        for(auto& item : ds.item_counts) {
            estimates[item.first] = item.second;
        }
    }

    static constexpr const char* prefix = "exact";
};

template <class T>
concept IsMockOracle = std::is_base_of<MockOracle, T>::value;

#endif