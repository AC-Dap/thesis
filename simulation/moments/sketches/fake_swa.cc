#include "moments/sketches/fake_swa.h"

#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <functional>
#include <cmath>

#include <iostream>

#include "common/mock_oracle.h"
#include "common/heap.h"
#include "common/utils/hashing.h"
#include "common/io/dataset.h"

using namespace std;

namespace moments {

void get_top_items(Heap<tuple<double, ItemId>>& h, const DatasetItemCounts& item_counts, const function<double(ItemId)>& weight_func) {
    if(h.cap == 0) return;

    for(ItemId item = 0; item < item_counts.size(); item++) {
        if (item_counts[item] == 0) continue;

        auto weight = weight_func(item);
        if(h.len < h.cap) {
            h.push({weight, item});
        } else if(weight > get<0>(h.heap[0])){
            h.pushpop({weight, item});
        }
    }
}

/**
 * We note that a SWA sample does not depend on the order of items received.
 * This means we can directly generate a sample without needing to simulate the data stream.
 */
tuple<vector<ItemId>, vector<double>, vector<double>> fake_swa_sample(
    size_t kh, size_t kp, size_t ku, size_t deg, MockOracle& oracle, Dataset& ds) {

    // Copy items, since we need to modify it.
    DatasetItemCounts item_counts_copy(ds.item_counts);

    // The items chosen, and their weights + probs
    vector<ItemId> s(kh + kp + ku);
    vector<double> weights(kh + kp + ku), probs(kh + kp + ku);

    random_device rd;
    mt19937 gen(rd());

    SeedFun seed = generate_seed_function(gen, ds.item_counts.size());

    // First, get top kh by oracle
    Heap<tuple<double, ItemId>> top_h(kh);
    get_top_items(top_h, item_counts_copy, [&](ItemId item) { return oracle.estimate(item); });

    // Add everything in heap to sample
    // Remove sampled items from `item_copy`
    for(int i = 0; i < kh; i++) {
        auto item = get<1>(top_h.heap[i]);
        s[i] = item;
        weights[i] = ds.item_counts[item];
        probs[i] = 1;
        item_counts_copy[item] = 0;
    }

    // Get top kp by weighted sample in remaining items
    Heap<tuple<double, ItemId>> top_p(kp + 1);
    get_top_items(top_p, item_counts_copy, [&](ItemId item) { return oracle.estimate(item) / pow(seed[item], 1./deg); });

    auto tau = get<0>(top_p.heap[0]);
    for(int i = 0; i < kp; i++) {
        auto item = get<1>(top_p.heap[1+i]);
        s[kh + i] = item;
        weights[kh + i] = ds.item_counts[item];
        probs[kh + i] = 1 - exp(-pow(oracle.estimate(item) / tau, deg));
        item_counts_copy[item] = 0;
    }

    // Get top ku by weighted sample in remaining items
    Heap<tuple<double, ItemId>> top_u(ku + 1);
    get_top_items(top_u, item_counts_copy, [&](ItemId item) { return -seed[item]; });

    tau = get<0>(top_u.heap[0]);
    for(int i = 0; i < ku; i++) {
        s[kh + kp + i] = get<1>(top_u.heap[1+i]);
        weights[kh + kp + i] = ds.item_counts[s[kh + kp + i]];
        probs[kh + kp + i] = 1 - exp(tau);
    }

    return {s, weights, probs};
}

}
