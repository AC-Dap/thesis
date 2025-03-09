#include "threshold/sketches/bucket_sketch.h"

#include <iostream>
#include <vector>
#include <functional>

#include "common/heap.h"
#include "common/bucket_sketch.h"
#include "common/utils/hashing.h"

using namespace std;

namespace threshold {

    double central_bucket_sketch(const size_t k_hh, const double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds) {
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
            double avg = (left + right) / 2.;
            double n_est = max(1., S / avg);

            estimate += avg >= threshold * N ? n_est : 0;
        }

        // Add top_hh
        for (int i = 0; i < top_h.len; i++) {
            estimate += ds.item_counts.at(get<1>(top_h.heap[i])) >= threshold * N ? 1 : 0;
        }

        return estimate;
    }

    double counting_bucket_sketch(size_t k_hh, const double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds){
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

            size_t S = bucket_counts[i];
            double n_est = bucket_p[i] / N;
            double avg = S / n_est;

            estimate += avg >= threshold * N ? n_est : 0;
        }

        // Add top_hh
        for (int i = 0; i < top_h.len; i++) {
            estimate += ds.item_counts.at(get<1>(top_h.heap[i])) >= threshold * N ? 1 : 0;
        }

        return estimate;
    }

    double sampling_bucket_sketch(size_t k_hh, size_t k_u, const double threshold, const Buckets& buckets, MockOracle& o, const Dataset& ds){
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
        size_t N = ds.lines.size();
        for (int bucket_i = 0; bucket_i < bucket_counts.size(); bucket_i++) {
            auto n_sampled = bucket_n_sampled[bucket_i];
            // Not enough elements, just add them
            if (n_sampled < k_u) {
                for (int i = 0; i < n_sampled; i++) {
                    estimate += ds.item_counts.at(bucket_p_sample[bucket_i][i]) >= threshold * N ? 1 : 0;
                }
            } else {
                // Take average of sampled elements
                double sum = 0;
                size_t n_over_threshold = 0;
                for (int i = 0; i < n_sampled; i++) {
                    auto item = bucket_p_sample[bucket_i][i];
                    sum += ds.item_counts[item];
                    n_over_threshold += ds.item_counts[item] >= threshold * N ? 1 : 0;
                }

                estimate += n_over_threshold * bucket_counts[bucket_i] / sum;
            }
        }

        // Add top_hh
        for (int i = 0; i < top_h.len; i++) {
            estimate += ds.item_counts.at(get<1>(top_h.heap[i])) >= threshold * N ? 1 : 0;
        }

        return estimate;
    }

}
