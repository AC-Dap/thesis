#include "bucket_sketch.h"

#include <hashing.h>
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

double bucket_sketch(const size_t k_hh, const size_t deg, const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector<size_t> bucket_counts(buckets.size() - 1, 0);
    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        auto freq_est = o.estimate(item);
        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += ds.item_counts[item];
    }, top_h, o, ds);

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

double smart_a_bucket_sketch(const size_t k_hh, const size_t deg, const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector<size_t> bucket_counts(buckets.size() - 1, 0);
    vector<pair<double, size_t>> bucket_min(buckets.size() - 1, {1, 0});
    vector<pair<double, size_t>> bucket_max(buckets.size() - 1, {0, 0});

    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        auto freq_est = o.estimate(item);
        auto S = ds.item_counts[item];

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
        if (freq_est < bucket_min[bucket_i - 1].first) {
            bucket_min[bucket_i - 1].first = freq_est;
            bucket_min[bucket_i - 1].second = S;
        }
        if (freq_est > bucket_max[bucket_i].first) {
            bucket_max[bucket_i].first = freq_est;
            bucket_max[bucket_i].second = S;
        }
    }, top_h, o, ds);

    // Get prediction of each bucket
    long double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        double S = bucket_counts[i];
        double left = bucket_min[i].second, right = bucket_max[i].second;
        double n_est = max(1., n_estimate(S, left, right));

        estimate += exp(deg * log(S) - (deg-1) * log(n_est));
    }

    // Add top_hh
    for (int i = 0; i < top_h.len; i++) {
        estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
    }

    return estimate;
}

double smart_b_bucket_sketch(const size_t k_hh, const size_t deg, const function<double(double, double, double)>& n_estimate, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector<size_t> bucket_counts(buckets.size() - 1, 0);
    vector<pair<double, size_t>> bucket_min(buckets.size() - 1, {1, 0});
    vector<pair<double, size_t>> bucket_max(buckets.size() - 1, {0, 0});

    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        auto freq_est = o.estimate(item);
        auto S = ds.item_counts[item];

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
        if (freq_est < bucket_min[bucket_i - 1].first) {
            bucket_min[bucket_i - 1].first = freq_est;
            bucket_min[bucket_i - 1].second = S;
        }
        if (freq_est > bucket_max[bucket_i].first) {
            bucket_max[bucket_i].first = freq_est;
            bucket_max[bucket_i].second = S;
        }
    }, top_h, o, ds);

    // Get prediction of each bucket
    long double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        double S = bucket_counts[i];
        double left = N * bucket_min[i].first, right = N * bucket_max[i].first;
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
    vector<long double> bucket_p3(buckets.size() - 1, 0); // For deg=4

    size_t N = ds.lines.size();

    // Take out the top k_hh items
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        auto freq_est = o.estimate(item);
        auto S = ds.item_counts[item];

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
        bucket_p[bucket_i - 1] += S * freq_est;
        bucket_p2[bucket_i - 1] += S * freq_est * freq_est;
        bucket_p3[bucket_i - 1] += S * freq_est * freq_est * freq_est;
    }, top_h, o, ds);

    // Get prediction of each bucket
    long double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        long double S = bucket_counts[i];
        long double Sp = bucket_p[i];
        long double Sp2 = bucket_p2[i];
        long double Sp3 = bucket_p3[i];
        if (deg == 3) {
            estimate += S +
                3 * (N-1) * Sp +
                    (N-1) * (N-2) * Sp2;
        } else if (deg == 4) {
            estimate += S +
                7 * (N-1) * Sp +
                6 * (N-1) * (N-2) * Sp2 +
                    (N-1) * (N-2) * (N-3) * Sp3;
        }
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
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        auto freq_est = o.estimate(item);
        auto S = ds.item_counts[item];

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        bucket_counts[bucket_i - 1] += S;
        bucket_p[bucket_i - 1] += S / freq_est;
    }, top_h, o, ds);

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

double swa_bucket_sketch(const size_t k_hh, const size_t k_p, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector bucket_top_p(buckets.size() - 1, Heap<tuple<double, ItemId>>(k_p + 1));

    random_device rd;
    mt19937 gen(rd());
    SeedFun seed = generate_seed_function(gen, ds);

    // Take out the top k_hh items
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        auto freq_est = o.estimate(item);

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        auto& h = bucket_top_p[bucket_i - 1];

        auto weight = freq_est / pow(seed[item], 1./deg);
        if(h.len < h.cap) {
            h.push({weight, item});
        } else if(weight > get<0>(h.heap[0])){
            h.pushpop({weight, item});
        }
    }, top_h, o, ds);

    // Get prediction of each bucket
    long double estimate = 0;
    for (auto& h : bucket_top_p) {
        if (h.len < h.cap) {
            for (int i = 0; i < h.len; i++) {
                estimate += pow(ds.item_counts.at(get<1>(h.heap[i])), deg);
            }
            continue;
        }

        auto tau = get<0>(h.heap[0]);
        for(int i = 0; i < h.len - 1; i++) {
            auto item = get<1>(h.heap[1+i]);
            auto weight = ds.item_counts[item];
            auto prob = 1 - exp(-pow(o.estimate(item) / tau, deg));
            estimate += pow(weight, deg) / prob;
        }
    }

    // Add top_hh
    for (int i = 0; i < top_h.len; i++) {
        estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
    }

    return estimate;
}

double unif_bucket_sketch(const size_t k_hh, const size_t k_p, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector bucket_top_p(buckets.size() - 1, Heap<tuple<double, ItemId>>(k_p));
    vector<size_t> bucket_counts(buckets.size() - 1, 0);

    random_device rd;
    mt19937 gen(rd());
    SeedFun seed = generate_seed_function(gen, ds);

    // Take out the top k_hh items
    Heap<tuple<double, ItemId>> top_h(k_hh);

    // Process items
    process_ds_to_buckets([&](ItemId item) {
        if (k_p == 0) return;

        auto freq_est = o.estimate(item);

        size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
        auto& h = bucket_top_p[bucket_i - 1];

        auto weight = -seed[item];
        if(h.len < h.cap) {
            h.push({weight, item});
        } else if(weight > get<0>(h.heap[0])){
            h.pushpop({weight, item});
        }

        bucket_counts[bucket_i - 1] += ds.item_counts[item];
    }, top_h, o, ds);

    // Get prediction of each bucket
    long double estimate = 0;
    for (int bucket_i = 0; bucket_i < bucket_top_p.size(); bucket_i++) {
        if (k_p == 0) continue;

        auto h = bucket_top_p[bucket_i];
        if (h.len < h.cap) {
            for (int i = 0; i < h.len; i++) {
                estimate += pow(ds.item_counts.at(get<1>(h.heap[i])), deg);
            }
            continue;
        }

        // auto tau = get<0>(h.heap[0]);
        // for(int i = 0; i < h.len - 1; i++) {
        //     auto item = get<1>(h.heap[1+i]);
        //     auto weight = ds.item_counts[item];
        //     auto prob = 1 - exp(tau);
        //     estimate += pow(weight, deg) / prob;
        // }
        double sum = 0;
        double est = 0;
        for (int i = 0; i < h.len; i++) {
            auto item = get<1>(h.heap[i]);
            sum += ds.item_counts[item];
            est += pow(ds.item_counts[item], deg);
        }
        estimate += est * bucket_counts[bucket_i] / sum;
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
