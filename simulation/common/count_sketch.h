#ifndef COUNT_SKETCH_H
#define COUNT_SKETCH_H

#include <tuple>
#include <vector>

#include "common/heap.h"
#include "common/utils/hashing.h"

using namespace std;

struct CountSketch {
    CountSketch(size_t width, size_t k, size_t num_unique_elements)
        : width(width), k(k), sketch(depth, vector<double>(width, 0)),
            top_k(k), hashes(depth), signs(depth), num_unique_elements(num_unique_elements) {
        reset_hashes();
    }

    void reset();
    void update(ItemId item, double count);
    double estimate(ItemId item);
    vector<ItemId> heavy_hitters() const;

    size_t space_size() const {
        return width * depth + k;
    }

    /**
     * Updates `hashes` and `signs` with new random hash and sign functions.
     */
    void reset_hashes() {
        std::random_device rd;
        std::mt19937 gen(rd());

        for(int i = 0; i < depth; i++) {
            hashes[i] = generate_hash_function(gen, num_unique_elements, width);
            signs[i] = generate_sign_function(gen, num_unique_elements);
        }
    }

    size_t width, k;
    static constexpr size_t depth = 7;    // Fix depth to 7 for optimization
    vector<vector<double>> sketch;
    Heap<tuple<double, ItemId>> top_k;

    vector<HashFun> hashes;
    vector<SignFun> signs;
    size_t num_unique_elements;
};

#endif