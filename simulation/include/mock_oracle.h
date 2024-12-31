#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <unordered_map>
#include <random>
#include <iostream>

#include "dataset.h"

using namespace std;

struct MockOracle {
    virtual ~MockOracle() = default;

    MockOracle(double ep, Dataset& ds): ep(ep), ds(ds) {
        estimates.reserve(ds.item_counts.size());
    }

    /**
     * Estimate is constrained to be at least 1.
     */
    double estimate(const string* item) const {
        const double N = ds.lines.size();

        if (const auto it = estimates.find(item); it != estimates.end()) {
            return max(1., it->second) / N;
        }
        return 1. / N;
    }

    virtual void reset_estimates(){};

    virtual string name() const = 0;

    double ep;
    unordered_map<const string*, double> estimates;

    Dataset& ds;
};

struct MockOracleRelativeError final : MockOracle {
    MockOracleRelativeError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(1.-ep, 1.+ep);

        for(auto& item : ds.item_counts) {
            estimates[item.first] = d(gen) * item.second;
        }
    }

    string name() const override {
        return "rel";
    }
};

struct MockOracleAbsoluteError final : MockOracle {
    MockOracleAbsoluteError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(-ep, ep);

        for(auto& item : ds.item_counts) {
            estimates[item.first] = item.second + d(gen) * ds.lines.size();
        }
    }

    string name() const override {
        return "abs";
    }
};

struct MockOracleBinomialError final : MockOracle {
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

    string name() const override {
        return "bin";
    }
};

struct ExactOracle final : MockOracle {
    explicit ExactOracle(Dataset& ds): MockOracle(0, ds){}

    void reset_estimates() override {
        for(auto& item : ds.item_counts) {
            estimates[item.first] = item.second;
        }
    }

    string name() const override {
        return "exact";
    }
};

template <class T>
concept IsMockOracle = std::is_base_of<MockOracle, T>::value;

#endif