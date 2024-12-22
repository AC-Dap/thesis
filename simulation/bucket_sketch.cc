#include "bucket_sketch.h"

#include <iostream>

#include "heap.h"

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

double bucket_sketch(const size_t k_hh, const size_t deg, const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector<size_t> bucket_counts(buckets.size() - 1, 0);
    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<double, const string*>> top_h(k_hh);

    // Place each item into the correct bucket
    for (auto& item : ds.item_counts) {
        // Constrain estimate between 1/N and 1
        double freq_est = min(1., max(1., o.estimate(item.first)) / N);
        size_t S = item.second;

        // First try and see if this is top_hh
        if (k_hh > 0) {
            if (top_h.len < top_h.cap) {
                top_h.push({freq_est, item.first});
                continue;
            }

            if (freq_est > get<0>(top_h.heap[0])) {
                auto old_item = top_h.pushpop({freq_est, item.first});
                freq_est = get<0>(old_item);
                S = ds.item_counts.at(get<1>(old_item));
            }
        }

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
    }

    // Get prediction of each bucket
    long double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        double left = buckets[i] * N, right = buckets[i + 1] * N;
        size_t S = bucket_counts[i];
        double n_est = max(1., n_estimate(S, left, right));

        estimate += exp(deg * log(S) - (deg-1) * log(n_est));
    }

    // Add top_hh
    for (int i = 0; i < top_h.len; i++) {
        estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
    }

    return estimate;
}

double cond_bucket_sketch(const size_t k_hh, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector<size_t> bucket_counts(buckets.size() - 1, 0);
    vector<long double> bucket_p(buckets.size() - 1, 0);
    vector<long double> bucket_p2(buckets.size() - 1, 0);

    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<long double, const string*>> top_h(k_hh);

    // Place each item into the correct bucket
    for (auto& item : ds.item_counts) {
        // Constrain estimate between 1/N and 1
        long double freq_est = min(1., max(1., o.estimate(item.first)) / N);
        size_t S = item.second;

        // First try and see if this is top_hh
        if (k_hh > 0) {
            if (top_h.len < top_h.cap) {
                top_h.push({freq_est, item.first});
                continue;
            }

            if (freq_est > get<0>(top_h.heap[0])) {
                auto old_item = top_h.pushpop({freq_est, item.first});
                freq_est = get<0>(old_item);
                S = ds.item_counts.at(get<1>(old_item));
            }
        }

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
        bucket_p[bucket_i - 1] += S * freq_est;
        bucket_p2[bucket_i - 1] += S * freq_est * freq_est;
    }

    // Get prediction of each bucket
    long double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        long double S = bucket_counts[i];
        long double Sp = bucket_p[i];
        long double Sp2 = bucket_p2[i];
        long double corr_factor = N/S;
        estimate += S + 3 * (S-1) * Sp * corr_factor + (S-1) * (S-2) * Sp2 * corr_factor * corr_factor;
    }

    // Add top_hh
    for (int i = 0; i < top_h.len; i++) {
        estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
    }

    return estimate;
}

double alt_bucket_sketch(const size_t k_hh, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector<size_t> bucket_counts(buckets.size() - 1, 0);
    vector<long double> bucket_p(buckets.size() - 1, 0);

    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<long double, const string*>> top_h(k_hh);

    // Place each item into the correct bucket
    for (auto& item : ds.item_counts) {
        // Constrain estimate between 1/N and 1
        long double freq_est = min(1., max(1., o.estimate(item.first)) / N);
        size_t S = item.second;

        // First try and see if this is top_hh
        if (k_hh > 0) {
            if (top_h.len < top_h.cap) {
                top_h.push({freq_est, item.first});
                continue;
            }

            if (freq_est > get<0>(top_h.heap[0])) {
                auto old_item = top_h.pushpop({freq_est, item.first});
                freq_est = get<0>(old_item);
                S = ds.item_counts.at(get<1>(old_item));
            }
        }

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
        bucket_p[bucket_i - 1] += S / freq_est;
    }

    // Get prediction of each bucket
    long double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        long double S = bucket_counts[i];
        long double log_n_est = log(bucket_p[i]) - log(N);
        estimate += exp(deg * log(S) - (deg-1) * log_n_est);
    }

    // Add top_hh
    for (int i = 0; i < top_h.len; i++) {
        estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
    }

    return estimate;
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
