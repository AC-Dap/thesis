#include "hashing.h"

#include <unordered_map>
#include <unordered_set>
#include <random>

#include "dataset.h"

using namespace std;

/*
    Generates a random hash for every item in `items`, mapping to an integer between
    0 and `width` exclusive.
 */
HashFun generate_hash_function(mt19937& rng, DatasetItems& items, size_t width) {
    std::uniform_int_distribution<> d(0, width - 1);
    HashFun hash;
    hash.reserve(items.size());

    for(auto item : items) {
        hash[item] = d(rng);
    }

    return hash;
}

/*
    Generates a random sign for every item in `items`.
 */
SignFun generate_sign_function(mt19937& rng, DatasetItems& items) {
    std::bernoulli_distribution d(0.5);
    SignFun hash;
    hash.reserve(items.size());

    for(auto item : items) {
        hash[item] = d(rng) ? 1 : -1;
    }

    return hash;
}

/*
    Generates a random seed for every item in `items`. The seed is distributed Expo(1).
 */
SeedFun generate_seed_function(mt19937& rng, DatasetItems& items) {
    std::exponential_distribution<> d(1);
    SeedFun hash;
    hash.reserve(items.size());

    for(auto item : items) {
        hash[item] = d(rng);
    }

    return hash;
}