#ifndef PPSWOR_H
#define PPSWOR_H

#include "count_sketch.h"
#include "hashing.h"

using namespace std;

struct PPSWOR {
    PPSWOR(size_t k, size_t deg, Dataset& ds):
        // CountSketch: ep = 1/(2k), width = e/ep
        k(k), deg(deg), cs(size_t(5.4366*k), k+1, ds), ds(ds) {
        reset_seed();
    }

    void update(ItemId item, double count);
    tuple<vector<ItemId>, vector<double>, vector<double>> sample();

    size_t space_size() const {
        return cs.space_size();
    }

    void reset() {
        cs.reset();
        reset_seed();
    }

    void reset_seed() {
        std::random_device rd;
        std::mt19937 gen(rd());

        seed = generate_seed_function(gen, ds);
        for(ItemId item = 0; item < ds.item_counts.size(); item++) {
            if (ds.item_counts[item] == 0) continue;
            
            seed[item] = pow(seed[item], 1./deg);
        }
    }

    size_t k, deg;
    CountSketch cs;

    SeedFun seed;
    Dataset& ds;
};

#endif