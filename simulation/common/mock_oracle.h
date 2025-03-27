#ifndef MOCK_ORACLE_H
#define MOCK_ORACLE_H

#include <string>
#include <random>
#include <vector>
#include <stdexcept>

#include "common/io/dataset.h"

using namespace std;

struct MockOracle {
    virtual ~MockOracle() = default;

    MockOracle(double ep, string name, Dataset& ds): ep(ep), name(name), estimates(ds.item_counts.size(), 0), ds(ds) {}

    /**
     * Returns an estimate of an item's frequency, constrained to be in [0, 1].
     */
    double estimate(ItemId item) const {
        if (item >= estimates.size()) {
            throw std::invalid_argument("item_id >= estimates.size()");
        }

        return estimates[item];
    }

    virtual void reset_estimates(){};

    double ep;
    string name;
    vector<double> estimates;

    Dataset& ds;
};

struct MockOracleRelativeError final : MockOracle {
    MockOracleRelativeError(double ep, string name, Dataset& ds): MockOracle(ep, name, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(1.-ep, 1.+ep);

        for(ItemId item = 0; item < estimates.size(); item++) {
            if (double count = ds.item_counts[item]; count > 0) {
                estimates[item] = d(gen) * count;
            } else {
                estimates[item] = 0;
            }
        }

        const size_t N = ds.lines.size();
        for(ItemId item = 0; item < estimates.size(); item++) {
            estimates[item] = min(1., max(0., estimates[item]) / N);
        }
    }
};

struct MockOracleAbsoluteError final : MockOracle {
    MockOracleAbsoluteError(double ep, string name, Dataset& ds): MockOracle(ep, name, ds) {}

    void reset_estimates() override {
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<> d(-ep, ep);

        size_t N = ds.lines.size();
        for(ItemId item = 0; item < estimates.size(); item++) {
            if (double count = ds.item_counts[item]; count > 0) {
                estimates[item] = count + d(gen) * N;
            } else {
                estimates[item] = 0;
            }
        }

        for(ItemId item = 0; item < estimates.size(); item++) {
            estimates[item] = min(1., max(0., estimates[item]) / N);
        }
    }
};

struct ExactOracle final : MockOracle {
    explicit ExactOracle(string name, Dataset& ds): MockOracle(0, name, ds){}

    void reset_estimates() override {
        for(ItemId item = 0; item < estimates.size(); item++) {
            if (size_t count = ds.item_counts[item]; count > 0) {
                estimates[item] = count;
            } else {
                estimates[item] = 1;
            }
        }

        const size_t N = ds.lines.size();
        for(ItemId item = 0; item < estimates.size(); item++) {
            estimates[item] = min(1., max(0., estimates[item]) / N);
        }
    }
};

#endif