#include "dataset.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

void Dataset::read_from_file(const string& path) {
    ifstream f(path);

    if(!f.is_open()) {
        throw std::invalid_argument("Unable to open dataset: " + path);
    }

    auto t1 = chrono::high_resolution_clock::now();

    vector<string> f_lines;
    string line;
    // Skip header
    getline(f, line);
    while(getline(f, line)) {
        size_t query_start = line.find('\t') + 1;
        size_t query_end = line.find('\t', query_start);
        string s = line.substr(query_start, query_end - query_start);

        if (!backing_items_.contains(s)) {
            throw std::invalid_argument("Entry \"" + s + "\" not found in backing items");
        }

        auto item_id = backing_items_[s];
        lines.push_back(item_id);
        item_counts[item_id]++;
    }
    f.close();

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Reading in file took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;
}

BackingItems get_backing_items(const vector<string>& file_names) {
    BackingItems backing_items;
    ItemId nextId = 0;

    for (const string& file_name : file_names) {
        ifstream f(file_name);

        if(!f.is_open()) {
            throw std::invalid_argument("Unable to open dataset: " + file_name);
        }

        string line;
        // Skip header
        getline(f, line);

        while(getline(f, line)) {
            size_t query_start = line.find('\t') + 1;
            size_t query_end = line.find('\t', query_start);
            string s = line.substr(query_start, query_end - query_start);

            if (!backing_items.contains(s)) {
                backing_items[s] = nextId;
                nextId++;
            }
        }
        f.close();
    }

    return backing_items;
}