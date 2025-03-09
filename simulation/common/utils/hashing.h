#ifndef HASHING_H
#define HASHING_H

#include <random>

#include "common/io/dataset.h"

using namespace std;

typedef vector<size_t> HashFun;
typedef vector<int> SignFun;
typedef vector<double> SeedFun;

/**
    Generates a random hash for `num_unique_elements`, mapping to an integer between
    0 and `width` exclusive.
 */
HashFun generate_hash_function(mt19937& rng, size_t num_unique_elements, size_t width);

/**
    Generates a random sign for `num_unique_elements`.
 */
SignFun generate_sign_function(mt19937& rng, size_t num_unique_elements);

/**
    Generates a random seed for `num_unique_elements`. The seed is distributed Expo(1).
 */
SeedFun generate_seed_function(mt19937& rng, size_t num_unique_elements);

#endif