#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <chrono>

using namespace std;

struct Dataset {
    vector<string> lines;
    unordered_set<string> items;
    unordered_map<string, size_t> item_counts;
};

bool read_dataset(const char* path, Dataset& ds) {
    ifstream f(path);

    if(!f.is_open()) {
        cerr << "Unable to open file" << endl;
        return false;
    }

    auto t1 = chrono::high_resolution_clock::now();

    string line;
    // Skip header
    getline(f, line);
    while(getline(f, line)) {
        size_t query_start = line.find('\t');
        size_t query_end = line.find('\t', query_start + 1);
        string s = line.substr(query_start, query_end - query_start);

        ds.lines.push_back(s);
        ds.items.insert(s);
        ds.item_counts[s]++;
    }
    f.close();

    auto t2 = chrono::high_resolution_clock::now();
    cout << "Reading in file took "
         << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() / 1000.0
         << " seconds" << endl;

    return true;
}

#endif