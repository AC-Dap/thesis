#ifndef HASHING_H
#define HASHING_H

#include <unordered_map>
#include <unordered_set>
#include <random>

#include "dataset.h"

using namespace std;

typedef unordered_map<const string*, size_t> HashFun;
typedef unordered_map<const string*, int> SignFun;
typedef unordered_map<const string*, double> SeedFun;

/*
    Generates a random hash for every item in `items`, mapping to an integer between
    0 and `width` exclusive.
 */
HashFun generate_hash_function(mt19937& rng, DatasetItems& items, size_t width);

/*
    Generates a random sign for every item in `items`.
 */
SignFun generate_sign_function(mt19937& rng, DatasetItems& items);

/*
    Generates a random seed for every item in `items`. The seed is distributed Expo(1).
 */
SeedFun generate_seed_function(mt19937& rng, DatasetItems& items);

#endif