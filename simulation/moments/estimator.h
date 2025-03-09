#ifndef MOMENTS_ESTIMATOR_H
#define MOMENTS_ESTIMATOR_H

#include <vector>
#include <cmath>

#include "common/io/dataset.h"

using namespace std;

namespace moments {
    __uint128_t calculate_exact_moment(Dataset& ds, size_t deg);
    long double estimate_moment(vector<double>& weights, vector<double>& probs, size_t deg);

    inline __uint128_t calculate_exact_moment(Dataset& ds, size_t deg) {
        __uint128_t sum = 0;
        for(size_t item_count : ds.item_counts) {
            // Manual integer power instead of floating-point pow()
            __uint128_t value = 1;
            for(size_t i = 0; i < deg; i++) {
                value *= item_count;
            }
            sum += value;
        }
        return sum;
    }

    inline long double estimate_moment(vector<double>& weights, vector<double>& probs, size_t deg) {
        // Use Kahan summation to avoid floating point errors
        long double sum = 0;
        long double c = 0.0;
        for(int i = 0; i < weights.size(); i++) {
            // Repeated squaring to avoid extra multiplications
            long double value = 1;
            long double exp = weights[i];
            for(int d = deg; d > 0; d = d >> 1) {
                if((d & 1) == 1) value *= exp;
                exp *= exp;
            }

            // Kahan summation algorithm
            long double term = value / probs[i];
            long double y = term - c;       // c is the running compensation
            long double t = sum + y;        // sum is big, y is small
            c = (t - sum) - y;        // (t - sum) cancels high-order bits of y
            // subtracting y recovers negative of low-order bits
            sum = t;                  // new sum
        }

        return sum;
    }

}

#endif