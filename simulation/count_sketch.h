#ifndef COUNT_SKETCH_H
#define COUNT_SKETCH_H

#include <tuple>
#include <vector>
#include <unordered_set>
#include <algorithm>

#include "heap.h"
#include "hashing.h"

using namespace std;

struct CountSketch {
    CountSketch(int width, int depth, int k, unordered_set<string>& items)
        : width(width), depth(depth), k(k), sketch(depth, vector<double>(width, 0)),
            _estimates(depth), top_k(k), hashes(depth), signs(depth), items(items) {
        reset_hashes();
    }

    void reset();
    void update(string& item, double count);
    double estimate(string& item);
    vector<string> heavy_hitters();

    inline size_t space_size() {
        return width * depth + k;
    }

    /**
     * Updates `hashes` and `signs` with new random hash and sign functions.
     */
    inline void reset_hashes() {
        std::random_device rd;
        std::mt19937 gen(rd());

        for(int i = 0; i < depth; i++) {
            hashes[i] = generate_hash_function(gen, items, width);
            signs[i] = generate_sign_function(gen, items);
        }
    }

    int width, depth, k;
    vector<vector<double>> sketch;
    Heap<tuple<double, string>> top_k;

    // Pre-allocate array for CountSketch::estimates to avoid many small allocations.
    vector<double> _estimates;

    vector<HashFun> hashes;
    vector<SignFun> signs;
    unordered_set<string>& items;
};

#endif