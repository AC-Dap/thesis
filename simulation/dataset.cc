#include "dataset.h"

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <chrono>

using namespace std;

bool read_dataset(const char* path, Dataset& ds) {
    ifstream f(path);

    if(!f.is_open()) {
        cerr << "Unable to open file" << endl;
        return false;
    }

    auto t1 = chrono::high_resolution_clock::now();

    vector<string> lines;
    string line;
    // Skip header
    getline(f, line);
    while(getline(f, line)) {
        size_t query_start = line.find('\t');
        size_t query_end = line.find('\t', query_start + 1);
        string s = line.substr(query_start, query_end - query_start);

        lines.push_back(s);
    }
    f.close();

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Reading in file took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

    // At this point, lines should be stable.
    // Now we want to work with const string* pointers, so we need
    // to get a pointer to one instance of every unique string.
    unordered_map<string, const string*> string_to_ptr;
    for(auto& line : lines) {
        string_to_ptr[line] = &line;
    }

    ds.lines.resize(lines.size());
    ds.items.reserve(string_to_ptr.size());
    ds.item_counts.reserve(string_to_ptr.size());

    for(auto& item : string_to_ptr) {
        ds.items.insert(item.second);
    }

    for(int i = 0; i < lines.size(); i++) {
        ds.lines[i] = string_to_ptr[lines[i]];
        ds.item_counts[ds.lines[i]]++;
    }

    return true;
}