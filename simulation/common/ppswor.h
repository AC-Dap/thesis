#ifndef PPSWOR_H
#define PPSWOR_H

#include "common/count_sketch.h"
#include "common/utils/hashing.h"

using namespace std;

struct PPSWOR {
    PPSWOR(size_t k, size_t deg, size_t num_unique_elements):
        // CountSketch: ep = 1/(2k), width = e/ep
        k(k/16), deg(deg), cs(size_t(15. * k / 16. / 7.), k/16 + 1, num_unique_elements), num_unique_elements(num_unique_elements) {
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

        seed = generate_seed_function(gen, num_unique_elements);
        for(ItemId item = 0; item < num_unique_elements; item++) {
            seed[item] = pow(seed[item], 1./deg);
        }
    }

    size_t k, deg;
    CountSketch cs;

    SeedFun seed;
    size_t num_unique_elements;
};

#endif