#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <vector>
#include <unordered_map>
#include <string>

#include "dataset.h"

using namespace std;

inline size_t exact_moment(DatasetItemCounts& item_counts, size_t deg) {
    size_t sum = 0;
    for(auto& item : item_counts) {
        sum += pow(item.second, deg);
    }
    return sum;
}

inline double estimate_moment(vector<double>& weights, vector<double>& probs, size_t deg) {
    double sum = 0;
    for(int i = 0; i < weights.size(); i++) {
        sum += pow(weights[i], deg) / probs[i];
    }

    return sum;
}

#endif