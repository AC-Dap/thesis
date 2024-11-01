#ifndef PPSWOR_H
#define PPSWOR_H

#include <unordered_set>
#include "count_sketch.h"
#include "hashing.h"

using namespace std;

struct PPSWOR {
    PPSWOR(size_t k, size_t deg, unordered_set<string>& items):
        k(k), deg(deg), cs(int(10/(0.05 * 0.05)), 7, k+1, items), items(items) {
        reset_seed();
    }

    void update(string& item, double count);
    tuple<vector<string>, vector<double>, vector<double>> sample();

    inline size_t space_size() {
        return cs.space_size();
    }

    inline void reset() {
        cs.reset();
        reset_seed();
    }

    inline void reset_seed() {
        std::random_device rd;
        std::mt19937 gen(rd());

        seed = generate_seed_function(gen, items);
        for(auto& item : items) {
            seed[item] = pow(seed[item], 1./deg);
        }
    }

    size_t k, deg;
    CountSketch cs;

    SeedFun seed;
    unordered_set<string>& items;
};

#endif