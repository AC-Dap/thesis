#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <random>
#include <iostream>
#include <vector>

#include "dataset.h"

using namespace std;

struct MockOracle {
    virtual ~MockOracle() = default;

    MockOracle(double ep, Dataset& ds): ep(ep), estimates(ds.item_counts.size(), 0), ds(ds) {}

    /**
     * Returns an estimate of an item's frequency, constrained to be in [1/N, 1].
     */
    double estimate(ItemId item) const {
        const size_t N = ds.lines.size();
        if (item >= estimates.size()) {
            throw std::invalid_argument("item_id >= estimates.size()");
        }

        return min(1., max(1., estimates[item]) / N);
    }

    virtual void reset_estimates(){};

    virtual string name() const = 0;

    double ep;
    vector<double> estimates;

    Dataset& ds;
};

struct MockOracleRelativeError final : MockOracle {
    MockOracleRelativeError(double ep, Dataset& ds): MockOracle(ep, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(1.-ep, 1.+ep);

        for(ItemId item = 0; item < estimates.size(); item++) {
            if (size_t count = ds.item_counts[item]; count > 0) {
                estimates[item] = d(gen) * count;
            }
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

        size_t N = ds.lines.size();
        for(ItemId item = 0; item < estimates.size(); item++) {
            if (size_t count = ds.item_counts[item]; count > 0) {
                estimates[item] = count + d(gen) * N;
            }
        }
    }

    string name() const override {
        return "abs";
    }
};

struct MockOracleBinomialError final : MockOracle {
    explicit MockOracleBinomialError(Dataset& ds): MockOracle(0, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        size_t N = ds.lines.size();
        for(ItemId item = 0; item < estimates.size(); item++) {
            if (size_t count = ds.item_counts[item]; count > 0) {
                binomial_distribution<> d(N, double(count)/N);
                estimates[item] = d(gen);
            }
        }
    }

    string name() const override {
        return "bin";
    }
};

struct ExactOracle final : MockOracle {
    explicit ExactOracle(Dataset& ds): MockOracle(0, ds){}

    void reset_estimates() override {
        for(ItemId item = 0; item < estimates.size(); item++) {
            estimates[item] = ds.item_counts[item];
        }
    }

    string name() const override {
        return "exact";
    }
};

template <class T>
concept IsMockOracle = std::is_base_of<MockOracle, T>::value;

#endif