#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

/**
 * Mapping from unique strings to ids.
 */
typedef size_t ItemId;
typedef unordered_map<string, ItemId> BackingItems;

typedef vector<ItemId> DatasetLines;
typedef vector<size_t> DatasetItemCounts;

struct Dataset {

    explicit Dataset(BackingItems& backing_items):
        backing_items_(backing_items), item_counts(backing_items.size(), 0) {}

    void read_from_file(const string& path);

    DatasetLines lines;
    DatasetItemCounts item_counts;

private:
    BackingItems& backing_items_;
};

BackingItems get_backing_items(const vector<string>& file_names);

#endif