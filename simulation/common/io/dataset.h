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

typedef vector<ItemId> DatasetLines;
typedef vector<size_t> DatasetItemCounts;

struct Dataset {

    explicit Dataset(size_t num_unique_items):
        item_counts(num_unique_items, 0) {}

    /**
    * Reads items from the file and adds them to the dataset by incrementing the count of each item.
    */
    void add_from_file(const string& path);

    /**
     * Clears the dataset.
     */
    void clear();

    DatasetLines lines;
    DatasetItemCounts item_counts;
};

size_t count_unique_items(vector<string> dataset_paths);

#endif