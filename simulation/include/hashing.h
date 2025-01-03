#ifndef HASHING_H
#define HASHING_H

#include <random>

#include "dataset.h"

using namespace std;

typedef vector<size_t> HashFun;
typedef vector<int> SignFun;
typedef vector<double> SeedFun;

/**
    Generates a random hash for every item in `ds`, mapping to an integer between
    0 and `width` exclusive.
 */
HashFun generate_hash_function(mt19937& rng, Dataset& ds, size_t width);

/**
    Generates a random sign for every item in `ds`.
 */
SignFun generate_sign_function(mt19937& rng, Dataset& ds);

/**
    Generates a random seed for every item in `ds`. The seed is distributed Expo(1).
 */
SeedFun generate_seed_function(mt19937& rng, Dataset& ds);

#endif