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
    bool read_from_file(const char* path);
    void print_top_k(size_t k);

    DatasetLines lines;
    DatasetItems items;
    DatasetItemCounts item_counts;

private:
    // Array of allocated strings to maintain string pointers
    unordered_set<string> _backing_items;
};

#endif