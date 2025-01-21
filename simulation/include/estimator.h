#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <vector>
#include <cmath>

#include "dataset.h"

using namespace std;

inline size_t calculate_exact_threshold(Dataset& ds, double threshold) {
    size_t count = 0;
    size_t threshould_count = threshold * ds.lines.size();
    for(size_t item_count : ds.item_counts) {
        if (item_count >= threshould_count) count++;
    }
    return count;
}

inline long double estimate_threshold(vector<double>& weights, vector<double>& probs, size_t threshold_count) {
    // Use Kahan summation to avoid floating point errors
    long double count = 0;
    for(int i = 0; i < weights.size(); i++) {
        if (weights[i] >= threshold_count && probs[i] != 0) count += 1 / probs[i];
    }

    return count;
}

#endif