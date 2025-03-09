#include "moments/sketches/parsimonious.h"

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

#include "common/io/dataset.h"
#include "common/mock_oracle.h"
#include "common/bucket_sketch.h"
#include "common/count_sketch.h"
#include "moments/estimator.h"

using namespace std;

namespace moments {

void remove_from_bucket(double freq_est, size_t count, Buckets& b, vector<size_t>& bucket_counts) {
    size_t bucket_i = lower_bound(b.begin(), b.end(), freq_est) - b.begin();
    if (bucket_i >= bucket_counts.size()) bucket_i = bucket_counts.size() - 1;  // If freq_est >= 1

    if (bucket_counts[bucket_i] > count) {
        bucket_counts[bucket_i] -= count;
    } else {
        bucket_counts[bucket_i] = 0;
    }
}

void add_to_bucket(double freq_est, size_t counts, Buckets& b, vector<size_t>& bucket_counts) {
    size_t bucket_i = lower_bound(b.begin(), b.end(), freq_est) - b.begin();
    if (bucket_i >= bucket_counts.size()) bucket_i = bucket_counts.size() - 1;  // If freq_est >= 1

    bucket_counts[bucket_i] += counts;
}

double parsimonious_bucket_sketch(size_t k, size_t deg, Buckets& b, MockOracle& o, Dataset& ds) {
    vector<size_t> exact_counts(ds.item_counts.size(), 0);
    vector<bool> queried(ds.item_counts.size(), false);
    vector<size_t> bucket_counts(b.size() - 1, 0);

    CountSketch cs(size_t(5.4366*k), k, ds.item_counts.size());

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> d(0, 1);

    size_t N = 0;
    for (ItemId item : ds.lines) {
        N++;

        // First, see if it is one of the items we've seen before
        // If so, update its exact count
        if (queried[item]) {
            exact_counts[item]++;
        } else {
            // Always remove from bucket sketch
            // If we're not querying, it'll be added back in with an updated freq_est
            double count_estimate = cs.estimate(item);
            double freq_est = count_estimate / N;
            remove_from_bucket(freq_est, count_estimate, b, bucket_counts);

            // Otherwise, ask rng to see if we should query oracle for this item
            double p = log(N) / N;
            if (d(gen) < p) {
                // If so, remove from CS sketch, add to exact sketch
                queried[item] = true;
                exact_counts[item] = count_estimate;

                // Remove from CS sketch
                cs.update(item, -count_estimate);
            } else {
                // Otherwise, add to CS sketch
                cs.update(item, 1);
            }
        }

        // Add to bucket sketch, using frequency estimation from exact/CS sketch
        // Only add if it's using CS sketch?
        if (!queried[item]) {
            double count_estimate = cs.estimate(item);
            double freq_est = count_estimate / N;
            add_to_bucket(freq_est, count_estimate, b, bucket_counts);
        }
    }

    double estimate = 0;
    for (size_t i = 0; i < bucket_counts.size(); i++) {
        if (bucket_counts[i] == 0) continue;

        double left = b[i] * N, right = b[i + 1] * N;
        size_t S = bucket_counts[i];
        double n_est = max(1., n_estimate_harm_avg(S, left, right));

        estimate += exp(deg * log(S) - (deg-1) * log(n_est));
    }

    for (ItemId item = 0; item < ds.item_counts.size(); item++) {
        if (!queried[item]) continue;
        cout << exact_counts[item] << " vs. " << ds.item_counts[item] << endl;
        estimate += pow(exact_counts[item], deg);
    }

    return estimate;
}

double parsimonious_swa_sketch(size_t kh, size_t kp, size_t k, size_t deg, MockOracle& o, Dataset& ds) {
    // Our sketch is composed of three parts:
    // 1. An exact-count top-kh sketch
    // 2. An oracle-based top-kp sketch
    // 3. A PPSWOR CS-based sketch for the overflow
    Heap<tuple<double, ItemId, size_t>> top_h(kh);
    Heap<tuple<double, ItemId, size_t>> top_p(kp + 1);
    CountSketch cs(size_t(5.4366*k), k, ds.item_counts.size());

    random_device rd;
    mt19937 gen(rd());

    SeedFun seed = generate_seed_function(gen, ds.item_counts.size());
    uniform_real_distribution<> d(0, 1);

    size_t N = 0;
    for (ItemId item : ds.lines) {
        N++;
        bool added = false;

        // See if the item already exists in the top-kh or top-kp sketch
        // If so, update the count
        for (int i = 0; i < top_h.len; i++) {
            auto h = top_h.heap[i];
            if (get<1>(h) == item) {
                get<2>(h)++;
                added = true;
            }
        }
        if (added) continue;

        for (int i = 0; i < top_p.len; i++) {
            auto p = top_p.heap[i];
            if (get<1>(p) == item) {
                get<2>(p)++;
                added = true;
            }
        }
        if (added) continue;

        // Otherwise, ask rng to see if we should query the oracle
        double p = 10. * log(N) / N;
        if (d(gen) < p) {
            // Query oracle -> see if we should add to top-kh or top-kp
            double freq_est = o.estimate(item);
            double count = cs.estimate(item) * pow(seed[item], 1. / deg) + 1;
            bool shouldAdd = true;

            // Remove from CS; we may re-add it back in if it's not in top-kh or top-kp
            cs.update(item, -cs.estimate(item));

            // First try adding to top_h
            // We may need to bump an item from top_h to top_p
            if (top_h.len < top_h.cap) {
                top_h.push({freq_est, item, count});
                shouldAdd = false;
            } else if (freq_est > get<0>(top_h.heap[0])) {
                auto overflow = top_h.pushpop({freq_est, item, count});
                item = get<1>(overflow);
                freq_est = get<0>(overflow);
                count = get<2>(overflow);
            }

            // Now, item could be the original or one we bumped from top_h
            if (shouldAdd) {
                // Reweight freq_est for PPSWOR sketch
                freq_est /= pow(seed[item], 1./deg);
                if (top_p.len < top_p.cap) {
                    top_p.push({freq_est, item, count});
                    shouldAdd = false;
                } else if (freq_est > get<0>(top_p.heap[0])) {
                    auto overflow = top_p.pushpop({freq_est, item, count});
                    item = get<1>(overflow);
                    count = get<2>(overflow);
                }
            }

            // Now, item could be the original, one we bumped from top_h, or one we bumped from top_p
            // Add to CS sketch
            if (shouldAdd) {
                cs.update(item, count / pow(seed[item], 1. / deg));
            }
        } else {
            // If no, add to CS sketch
            cs.update(item, 1. / pow(seed[item], 1. / deg));
        }
    }

    vector<double> weights, probs;

    // Add exact counts from top_kh
    cout << top_h.len << endl;
    for(int i = 0; i < top_h.len; i++) {
        auto count = get<2>(top_h.heap[i]);
        weights.push_back(count);
        probs.push_back(1);
    }

    // Add weighted estimate from top_kp
    cout << top_p.len << endl;
    if (top_p.len > 0) {
        auto tau = get<0>(top_p.heap[0]);
        for(int i = 0; i < top_p.len - 1; i++) {
            auto freq_est = get<0>(top_p.heap[1+i]);
            auto count = get<2>(top_p.heap[1+i]);
            weights.push_back(count);
            probs.push_back(1 - exp(-pow(freq_est / tau, deg)));
        }
    }

    // Add weighted estimate from CS sketch
    // Normally, we would maintain this heap throughout the algorithm.
    // To simplify simulation, we just build it at the end.
    Heap<double> hh(k + 1);
    for (ItemId id = 0; id < ds.item_counts.size(); id++) {
        if (ds.item_counts[id] == 0) continue;

        auto freq_est = cs.estimate(id);
        auto count = freq_est * pow(seed[id], 1. / deg);
        if (hh.len < hh.cap) {
            hh.push(count);
        } else if (count > hh.heap[0]) {
            hh.pushpop(count);
        }
    }

    auto tau = hh.heap[0];
    for(int i = 0; i < k; i++){
        auto sample_count = hh.heap[1 + i];
        auto sample_prob = 1 - exp(-pow(sample_count/tau, deg));

        weights.push_back(sample_count);
        probs.push_back(sample_prob);
    }

    return estimate_moment(weights, probs, deg);
}

}