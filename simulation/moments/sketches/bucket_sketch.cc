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
            if (bucket_i == 0) cout << "Warning: bucket_i is 0: " << freq_est << endl;
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

}
