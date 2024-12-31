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

/**
 * Set of allocated strings to maintain string pointers
 */
typedef unordered_set<string> BackingItems;

struct Dataset {

    explicit Dataset(BackingItems& backing_items): backing_items_(backing_items) {}

    void read_from_file(const char* path);
    void print_top_k(size_t k);

    DatasetLines lines;
    DatasetItems items;
    DatasetItemCounts item_counts;

private:
    BackingItems& backing_items_;
};

BackingItems get_backing_items(vector<string> file_names);

#endif