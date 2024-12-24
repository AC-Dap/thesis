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

bool Dataset::read_from_file(const char* path) {
    ifstream f(path);

    if(!f.is_open()) {
        cerr << "Unable to open file" << endl;
        return false;
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

        f_lines.push_back(s);
    }
    f.close();

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Reading in file took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

    // At this point, lines should be stable.
    // Now we want to work with const string* pointers, so we need
    // to collect all unique strings
    for(auto& line : f_lines) {
        _backing_items.insert(line);
    }

    lines.resize(f_lines.size());
    items.reserve(_backing_items.size());
    item_counts.reserve(_backing_items.size());

    for(auto& item : _backing_items) {
        items.insert(&item);
    }

    for(int i = 0; i < f_lines.size(); i++) {
        lines[i] = &*_backing_items.find(f_lines[i]);
        item_counts[lines[i]]++;
    }

    return true;
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

    cout << endl << "top 10" << endl;

    for(auto [fst, snd] : sorted)
    {
        cout << *fst << "  " << snd << endl;
    }
}