#include "moments/sketches/bucket_sketch.h"

#include <iostream>
#include <vector>
#include <functional>

#include "common/heap.h"
#include "common/bucket_sketch.h"
#include "common/utils/hashing.h"

using namespace std;

namespace moments {

    double central_bucket_sketch(const size_t k_hh, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
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
            estimate += S * pow((left + right) / 2., deg - 1);
        }

        // Add top_hh
        for (int i = 0; i < top_h.len; i++) {
            estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
        }

        return estimate;
    }

    double unbiased_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds){
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

    double counting_bucket_sketch(size_t k_hh, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds){
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

    double sampling_bucket_sketch(size_t k_hh, size_t k_u, size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds){
        vector bucket_p_sample(buckets.size() - 1, vector<ItemId>(k_u));
        vector<size_t> bucket_n_sampled(buckets.size() - 1, 0);
        vector<size_t> bucket_counts(buckets.size() - 1, 0);

        // Take out the top k_hh items
        Heap<tuple<double, ItemId>> top_h(k_hh);

        // Process items
        process_ds_to_buckets([&](ItemId item) {
            if (k_u == 0) return;

            auto freq_est = o.estimate(item);

            size_t bucket_i = lower_bound(buckets.begin(), buckets.end(), freq_est) - buckets.begin();
            bucket_counts[bucket_i - 1] += ds.item_counts[item];
            if (bucket_n_sampled[bucket_i - 1] < k_u) {
                bucket_p_sample[bucket_i - 1][bucket_n_sampled[bucket_i - 1]++] = item;
            }
        }, top_h, o, ds);

        // Get prediction of each bucket
        long double estimate = 0;
        for (int bucket_i = 0; bucket_i < bucket_counts.size(); bucket_i++) {
            auto n_sampled = bucket_n_sampled[bucket_i];
            // Not enough elements, just add them
            if (n_sampled < k_u) {
                for (int i = 0; i < n_sampled; i++) {
                    estimate += pow(ds.item_counts.at(bucket_p_sample[bucket_i][i]), deg);
                }
            } else {
                // Take average of sampled elements
                double sum = 0;
                for (int i = 0; i < n_sampled; i++) {
                    auto item = bucket_p_sample[bucket_i][i];
                    sum += ds.item_counts[item];
                }

                estimate += bucket_counts[bucket_i] * pow(sum / n_sampled, deg - 1);
            }
        }

        // Add top_hh
        for (int i = 0; i < top_h.len; i++) {
            estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
        }

        return estimate;
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

double swa_bucket_sketch(const size_t k_hh, const size_t k_p, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector bucket_top_p(buckets.size() - 1, Heap<tuple<double, ItemId>>(k_p + 1));

    random_device rd;
    mt19937 gen(rd());
    SeedFun seed = generate_seed_function(gen, ds.item_counts.size());

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

    random_device rd;
    mt19937 gen(rd());
    SeedFun seed = generate_seed_function(gen, ds.item_counts.size());

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

        auto tau = get<0>(h.heap[0]);
        for(int i = 0; i < h.len - 1; i++) {
            auto item = get<1>(h.heap[1+i]);
            auto weight = ds.item_counts[item];
            auto prob = 1 - exp(tau);
            estimate += pow(weight, deg) / prob;
        }
    }

    // Add top_hh
    for (int i = 0; i < top_h.len; i++) {
        estimate += pow(ds.item_counts.at(get<1>(top_h.heap[i])), deg);
    }

    return estimate;
}

double unif2_bucket_sketch(const size_t k_hh, const size_t k_p, const size_t deg, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
    vector bucket_top_p(buckets.size() - 1, Heap<tuple<double, ItemId>>(k_p));
    vector<size_t> bucket_counts(buckets.size() - 1, 0);

    random_device rd;
    mt19937 gen(rd());
    SeedFun seed = generate_seed_function(gen, ds.item_counts.size());

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

}
