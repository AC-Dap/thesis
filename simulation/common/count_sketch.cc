#include "common/count_sketch.h"

#include <tuple>
#include <vector>
#include <algorithm>
#include <array>

#include "common/heap.h"
#include "common/utils/hashing.h"

using namespace std;

void CountSketch::reset() {
    for(auto& row : sketch) {
        ranges::fill(row, 0);
    }
    top_k.len = 0;

    reset_hashes();
}

void CountSketch::update(ItemId item, double count) {
    // Used later to guess if item is already in top_k or not
    double old_est = estimate(item);

    // Update sketch counts
    for(int i = 0; i < depth; i++) {
        size_t j = hashes[i][item];
        int s = signs[i][item];
        sketch[i][j] += s * count;
    }

    // Update top-k heap
    double new_est = estimate(item);

    // Check if item is already in top_k
    if(top_k.len > 0 && old_est >= get<0>(top_k.heap[0])) {
        for(int i = 0; i < top_k.len; i++) {
            if(get<1>(top_k.heap[i]) == item) {
                get<0>(top_k.heap[i]) = new_est;
                top_k.heapify(i);
                return;
            }
        }
    }

    // Item is not, so we pushpop to maintain top_k
    if(top_k.len < k) {
        top_k.push({new_est, item});
        return;
    }

    if(new_est > get<0>(top_k.heap[0])) {
        top_k.pushpop({new_est, item});
    }
}

double CountSketch::estimate(ItemId item) {
    // Get estimates from each row
    array<double, depth> estimates{};
    for(int i = 0; i < depth; i++) {
        size_t j = hashes[i][item];
        int s = signs[i][item];
        estimates[i] = s * sketch[i][j];
    }

    // Return median
    size_t n = depth / 2;
    ranges::nth_element(estimates, estimates.begin() + n);
    return estimates[n];
}

vector<ItemId> CountSketch::heavy_hitters() const {
    vector<ItemId> hh(k);
    for(int i = 0; i < top_k.len; i++) {
        hh[i] = get<1>(top_k.heap[i]);
    }

    return hh;
}
