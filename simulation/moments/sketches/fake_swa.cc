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

void get_top_items(Heap<tuple<double, ItemId>>& h, const vector<bool>& include_item, const function<double(ItemId)>& weight_func) {
    if(h.cap == 0) return;

    for(ItemId item = 0; item < include_item.size(); item++) {
        if (!include_item[item]) continue;

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

    // Whether we include items
    // To start, include all items with positive counts
    vector<bool> include_item(ds.item_counts.size(), false);
    for (ItemId item = 0; item < include_item.size(); item++) {
        include_item[item] = ds.item_counts[item] > 0;
    }

    // The items chosen, and their weights + probs
    vector<ItemId> s;
    vector<double> weights, probs;

    random_device rd;
    mt19937 gen(rd());

    SeedFun seed = generate_seed_function(gen, ds.item_counts.size());

    // First, get top kh by oracle
    Heap<tuple<double, ItemId>> top_h(kh);
    get_top_items(top_h, include_item, [&](ItemId item) { return oracle.estimate(item); });

    // Add everything in heap to sample
    // Remove sampled items from `item_copy`
    for(int i = 0; i < top_h.len; i++) {
        auto item = get<1>(top_h.heap[i]);
        s.push_back(item);
        weights.push_back(ds.item_counts[item]);
        probs.push_back(1);
        include_item[item] = false;
    }

    // Get top kp by weighted sample in remaining items
    // Change sample to not include anything where o(x) == 0
    vector<bool> include_item_kp(include_item);
    for (ItemId item = 0; item < include_item.size(); item++) {
        if (oracle.estimate(item) == 0) {
            include_item_kp[item] = false;
        }
    }

    Heap<tuple<double, ItemId>> top_p(kp + 1);
    get_top_items(top_p, include_item_kp, [&](ItemId item) { return oracle.estimate(item) / pow(seed[item], 1./deg); });

    if (top_p.len > 0) {
    	auto tau = get<0>(top_p.heap[0]);
    	for(int i = 0; i < top_p.len - 1; i++) {
        	auto item = get<1>(top_p.heap[1+i]);
        	s.push_back(item);
        	weights.push_back(ds.item_counts[item]);
        	probs.push_back(1 - exp(-pow(oracle.estimate(item) / tau, deg)));
    	}
    }

    // Get top ku by weighted sample for those estimates with o(x) == 0
    vector<bool> include_item_ku(include_item);
    for (ItemId item = 0; item < include_item.size(); item++) {
        if (oracle.estimate(item) > 0) {
            include_item_ku[item] = false;
        }
    }

    Heap<tuple<double, ItemId>> top_u(ku + 1);
    get_top_items(top_u, include_item_ku, [&](ItemId item) { return -seed[item]; });

    if (top_u.len > 0) {
    	auto tau = get<0>(top_u.heap[0]);
    	for(int i = 0; i < top_u.len - 1; i++) {
        	auto item = get<1>(top_u.heap[1+i]);
        	s.push_back(item);
        	weights.push_back(ds.item_counts[item]);
        	probs.push_back(1 - exp(tau));
    	}
    }

    return {s, weights, probs};
}

}
