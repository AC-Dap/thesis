#include "common/io/dataset.h"

#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <unordered_set>

using namespace std;

void Dataset::add_from_file(const string& path) {
    ifstream f(path);

    if(!f.is_open()) {
        throw std::invalid_argument("Unable to open dataset: " + path);
    }

    string line;
    while(getline(f, line)) {
        auto item_id = std::stoi(line);
        lines.push_back(item_id);
        item_counts[item_id]++;
    }
    f.close();
}

void Dataset::clear() {
    lines.clear();
    ranges::fill(item_counts, 0);
}

size_t count_unique_items(vector<string> dataset_paths) {
    ItemId maxId = 0;

    for (auto& path : dataset_paths) {
        ifstream f(path);

        if(!f.is_open()) {
            throw std::invalid_argument("Unable to open dataset: " + path);
        }

        string line;
        while(getline(f, line)) {
            auto item_id = std::stoi(line);
            if (item_id > maxId) maxId = item_id;
        }
        f.close();
    }

    return maxId + 1;
}