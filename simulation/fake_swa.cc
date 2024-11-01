#include "fake_swa.h"

#include <unordered_set>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <functional>

#include <iostream>

#include "mock_oracle.h"
#include "heap.h"
#include "hashing.h"

using namespace std;

void get_top_items(Heap<tuple<double, string>>& h, unordered_set<string>& items, function<double(string const&)> weight_func) {
    if(h.cap == 0) return;

    for(auto& item : items) {
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
tuple<vector<string>, vector<double>, vector<double>> fake_swa_sample(
    size_t kh, size_t kp, size_t ku, size_t deg, MockOracle& oracle,
    unordered_set<string>& items, unordered_map<string, size_t>& item_counts) {

    // Copy items, since we need to modify it.
    unordered_set<string> item_copy(items);

    vector<string> s(kh + kp + ku);
    vector<double> weights(kh + kp + ku), probs(kh + kp + ku);

    random_device rd;
    mt19937 gen(rd());

    SeedFun seed = generate_seed_function(gen, item_copy);

    // First, get top kh by oracle
    Heap<tuple<double, string>> top_h(kh);
    get_top_items(top_h, item_copy, [&](string const& item) { return oracle.estimate(item); });

    // Add everything in heap to sample
    // Remove sampled items from `item_copy`
    for(int i = 0; i < kh; i++) {
        auto item = get<1>(top_h.heap[i]);
        s[i] = item;
        weights[i] = item_counts[item];
        probs[i] = 1;
        item_copy.erase(item);
    }

    // Get top kp by weighted sample in remaining items
    Heap<tuple<double, string>> top_p(kp + 1);
    get_top_items(top_p, item_copy, [&](string const& item) { return pow(oracle.estimate(item), deg) / seed[item]; });

    auto tau = get<0>(top_p.heap[0]);
    for(int i = 0; i < kp; i++) {
        auto item = get<1>(top_p.heap[1+i]);
        s[kh + i] = item;
        weights[kh + i] = item_counts[item];
        probs[kh + i] = 1 - exp(-pow(oracle.estimate(item), deg) / tau);
        item_copy.erase(item);
    }

    // Get top ku by weighted sample in remaining items
    Heap<tuple<double, string>> top_u(ku + 1);
    get_top_items(top_u, item_copy, [&](string const& item) { return -seed[item]; });

    tau = get<0>(top_u.heap[0]);
    for(int i = 0; i < ku; i++) {
        s[kh + kp + i] = get<1>(top_u.heap[1+i]);
        weights[kh + kp + i] = item_counts[s[kh + kp + i]];
        probs[kh + i] = 1 - exp(-tau);
    }

    return {s, weights, probs};
}
