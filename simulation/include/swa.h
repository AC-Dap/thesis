#ifndef SWA_H
#define SWA_H

#include <vector>
#include <tuple>

#include "mock_oracle.h"
#include "hashing.h"
#include "heap.h"

using namespace std;

struct SWA {
    SWA(size_t kh, size_t kp, size_t ku, size_t deg, MockOracle& oracle, Dataset& ds):
        kh(kh), kp(kp + 1), ku(ku + 1), deg(deg), oracle(oracle), ds(ds),
        h_heap(kh), p_heap(kp), u_heap(ku) {

        reset_hashes();
    }

    void update(ItemId item, size_t count);
    tuple<vector<ItemId>, vector<double>, vector<double>> sample();

    size_t space_size() const {
        return kh + kp + ku;
    }

    void reset() {
        h_heap.len = p_heap.len = u_heap.len = 0;
        reset_hashes();
    }

    void reset_hashes() {
        std::random_device rd;
        std::mt19937 gen(rd());

        seed = generate_seed_function(gen, ds);
    }

    size_t kh, kp, ku, deg;
    Heap<tuple<double, ItemId, size_t>> h_heap, p_heap, u_heap;

    MockOracle& oracle;
    SeedFun seed;
    Dataset& ds;
};

#endif