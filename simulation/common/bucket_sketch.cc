#include "common/bucket_sketch.h"

#include "common/utils/hashing.h"
#include <iostream>

#include "common/heap.h"

Buckets generate_exponential_buckets(double min_freq, size_t k) {
    double step_size = min_freq / k;
    Buckets buckets(k + 1, 0);
    for (int i = 0; i < k; i++) {
        buckets[i] = pow(10., i * -step_size);
    }
    reverse(buckets.begin(), buckets.end());
    return buckets;
}

Buckets generate_linear_buckets(size_t k) {
    Buckets buckets(k+1, 1);
    for (int i = 0; i < k; i++) {
        buckets[i] = i * 1. / k;
    }
    return buckets;
}

void process_ds_to_buckets(const function<void(ItemId)> process_item, Heap<tuple<double, ItemId>>& top_h, MockOracle& o, const Dataset& ds) {
    // Place each item into the correct bucket
    for (ItemId item = 0; item < ds.item_counts.size(); item++) {
        if (ds.item_counts[item] == 0) continue;

        ItemId curr_item = item;
        double freq_est = o.estimate(item);

        // First try and see if we want to store anything in top_h
        if (top_h.cap > 0) {
            if (top_h.len < top_h.cap) {
                top_h.push({freq_est, item});
                continue;
            }

            if (freq_est > get<0>(top_h.heap[0])) {
                auto old_item = top_h.pushpop({freq_est, item});
                curr_item = get<1>(old_item);
            }
        }

        process_item(curr_item);
    }
}

double n_estimate_left(double S, double left, double right) {
    return std::ceil(S / left);
}

double n_estimate_right(double S, double left, double right) {
    return std::floor(S / right);
}

double n_estimate_left_round(double S, double left, double right) {
    return std::ceil(S / std::ceil(left));
}

double n_estimate_right_round(double S, double left, double right) {
    return std::floor(S / std::floor(right));
}

double n_estimate_arith_avg(double S, double left, double right) {
    return std::round(S * (left + right) / (2 * left * right));
}

double n_estimate_geo_avg(double S, double left, double right) {
    return std::round(S / std::sqrt(left * right));
}

double n_estimate_harm_avg(double S, double left, double right) {
    return std::round(2 * S / (left + right));
}
