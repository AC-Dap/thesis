#include "dataset.h"

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

using namespace std;

void Dataset::read_from_file(const char* path) {
    ifstream f(path);

    if(!f.is_open()) {
        throw std::invalid_argument("Unable to open file");
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

        auto query_ptr = backing_items_.find(s);
        if (query_ptr == backing_items_.end()) {
            throw std::invalid_argument("Entry \"" + s + "\" not found in backing items");
        }

        lines.push_back(&*query_ptr);
        items.insert(&*query_ptr);
        item_counts[&*query_ptr]++;
    }
    f.close();

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Reading in file took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;
}


void Dataset::print_top_k(size_t k) {
    // create container for top 10
    vector<pair<const string*, int>> sorted(k);

    // sort and copy with reversed compare function using second value of std::pair
    partial_sort_copy(item_counts.begin(), item_counts.end(),
                      sorted.begin(), sorted.end(),
                      [](const pair<const string*, int> &a, const pair<const string*, int> &b)
    {
        return a.second >= b.second;
    });

    cout << endl << "top " << k << endl;

    for(auto [fst, snd] : sorted)
    {
        cout << "\"" << *fst << "\" (" << fst << "): " << snd << endl;
    }
}

BackingItems get_backing_items(vector<string> file_names) {
    BackingItems backing_items;
    for (const string& file_name : file_names) {
        ifstream f(file_name);

        if(!f.is_open()) {
            throw std::invalid_argument("Unable to open file");
        }

        string line;
        // Skip header
        getline(f, line);
        while(getline(f, line)) {
            size_t query_start = line.find('\t') + 1;
            size_t query_end = line.find('\t', query_start);
            string s = line.substr(query_start, query_end - query_start);

            backing_items.insert(s);
        }
        f.close();
    }

    return backing_items;
}