#ifndef COUNT_SKETCH_H
#define COUNT_SKETCH_H

#include <tuple>
#include <vector>
#include <unordered_set>

#include "heap.h"
#include "hashing.h"

using namespace std;

struct CountSketch {
    CountSketch(size_t width, size_t k, Dataset& ds)
        : width(width), k(k), sketch(depth, vector<double>(width, 0)),
            top_k(k), hashes(depth), signs(depth), ds(ds) {
        reset_hashes();
    }

    void reset();
    void update(const string* item, double count);
    double estimate(const string* item);
    vector<const string*> heavy_hitters();

    size_t space_size() {
        return width * depth + k;
    }

    /**
     * Updates `hashes` and `signs` with new random hash and sign functions.
     */
    void reset_hashes() {
        std::random_device rd;
        std::mt19937 gen(rd());

        for(int i = 0; i < depth; i++) {
            hashes[i] = generate_hash_function(gen, ds.items, width);
            signs[i] = generate_sign_function(gen, ds.items);
        }
    }

    size_t width, k;
    static constexpr size_t depth = 7;    // Fix depth to 7 for optimization
    vector<vector<double>> sketch;
    Heap<tuple<double, const string*>> top_k;

    vector<HashFun> hashes;
    vector<SignFun> signs;
    Dataset& ds;
};

#endif