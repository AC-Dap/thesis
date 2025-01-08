#include "hashing.h"

#include <random>

#include "dataset.h"

using namespace std;

/**
    Generates a random hash for every item in `ds`, mapping to an integer between
    0 and `width` exclusive.
 */
HashFun generate_hash_function(mt19937& rng, Dataset& ds, size_t width) {
    std::uniform_int_distribution<> d(0, width - 1);
    HashFun hash(ds.item_counts.size());

    for(ItemId item = 0; item < ds.item_counts.size(); item++) {
        if(ds.item_counts[item] == 0) continue;
        hash[item] = d(rng);
    }

    return hash;
}

/**
    Generates a random sign for every item in `ds`.
 */
SignFun generate_sign_function(mt19937& rng, Dataset& ds) {
    std::bernoulli_distribution d(0.5);
    SignFun hash(ds.item_counts.size());

    for(ItemId item = 0; item < ds.item_counts.size(); item++) {
        if(ds.item_counts[item] == 0) continue;
        hash[item] = d(rng) ? 1 : -1;
    }

    return hash;
}

/**
    Generates a random seed for every item in `ds`. The seed is distributed Expo(1).
 */
SeedFun generate_seed_function(mt19937& rng, const Dataset& ds) {
    std::exponential_distribution<> d(1);
    SeedFun hash(ds.item_counts.size());

    for(ItemId item = 0; item < ds.item_counts.size(); item++) {
        if(ds.item_counts[item] == 0) continue;
        hash[item] = d(rng);
    }

    return hash;
}