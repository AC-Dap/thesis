#ifndef SWA_H
#define SWA_H

#include <vector>
#include <tuple>
#include <unordered_set>

#include "mock_oracle.h"
#include "hashing.h"
#include "heap.h"

using namespace std;

struct SWA {
    SWA(size_t kh, size_t kp, size_t ku, size_t deg, MockOracle oracle, Dataset& ds):
        kh(kh), kp(kp + 1), ku(ku + 1), deg(deg), oracle(oracle), ds(ds),
        h_heap(kh), p_heap(kp), u_heap(ku) {

        reset_hashes();
    }

    void update(const string* item, size_t count);
    tuple<vector<const string*>, vector<double>, vector<double>> sample();

    inline size_t space_size() {
        return kh + kp + ku;
    }

    inline void reset() {
        h_heap.len = p_heap.len = u_heap.len = 0;
        reset_hashes();
    }

    inline void reset_hashes() {
        std::random_device rd;
        std::mt19937 gen(rd());

        seed = generate_seed_function(gen, ds.items);
    }

    size_t kh, kp, ku, deg;
    Heap<tuple<double, const string*, size_t>> h_heap, p_heap, u_heap;

    MockOracle oracle;
    SeedFun seed;
    Dataset& ds;
};

#endif