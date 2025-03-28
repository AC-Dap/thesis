#include "hashing.h"

#include <random>

#include "common/io/dataset.h"

using namespace std;

/**
    Generates a random hash for every item in `ds`, mapping to an integer between
    0 and `width` exclusive.
 */
HashFun generate_hash_function(mt19937& rng, size_t num_unique_elements, size_t width) {
    std::uniform_int_distribution<> d(0, width - 1);
    HashFun hash(num_unique_elements);

    for(ItemId item = 0; item < num_unique_elements; item++) {
        hash[item] = d(rng);
    }

    return hash;
}

/**
    Generates a random sign for every item in `ds`.
 */
SignFun generate_sign_function(mt19937& rng, size_t num_unique_elements) {
    std::bernoulli_distribution d(0.5);
    SignFun hash(num_unique_elements);

    for(ItemId item = 0; item < num_unique_elements; item++) {
        hash[item] = d(rng) ? 1 : -1;
    }

    return hash;
}

/**
    Generates a random seed for every item in `ds`. The seed is distributed Expo(1).
 */
SeedFun generate_seed_function(mt19937& rng, size_t num_unique_elements) {
    std::exponential_distribution<> d(1);
    SeedFun hash(num_unique_elements);

    for(ItemId item = 0; item < num_unique_elements; item++) {
        hash[item] = d(rng);
    }

    return hash;
}