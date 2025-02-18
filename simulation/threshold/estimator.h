#ifndef THRESHOLD_ESTIMATOR_H
#define THRESHOLD_ESTIMATOR_H

#include <vector>
#include <cmath>

#include "common/io/dataset.h"

using namespace std;

namespace threshold {
    size_t calculate_exact_threshold(const Dataset& ds, double threshold);
    long double estimate_threshold(const vector<double>& weights, const vector<double>& probs, size_t threshold_count);

    inline size_t calculate_exact_threshold(const Dataset& ds, double threshold) {
        size_t count = 0;
        size_t threshold_count = threshold * ds.lines.size();
        for(size_t item_count : ds.item_counts) {
            if (item_count >= threshold_count) count++;
        }
        return count;
    }

    inline long double estimate_threshold(const vector<double>& weights, const vector<double>& probs, size_t threshold_count) {
        // Use Kahan summation to avoid floating point errors
        long double count = 0;
        for(size_t i = 0; i < weights.size(); i++) {
            if (weights[i] >= threshold_count && probs[i] != 0) count += 1 / probs[i];
        }

        return count;
    }

}

#endif