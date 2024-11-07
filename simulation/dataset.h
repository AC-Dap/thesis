#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

using namespace std;

typedef vector<const string*> DatasetLines;
typedef unordered_set<const string*> DatasetItems;
typedef unordered_map<const string*, size_t> DatasetItemCounts;

struct Dataset {
    DatasetLines lines;
    DatasetItems items;
    DatasetItemCounts item_counts;
};

bool read_dataset(const char* path, Dataset& ds);

#endif