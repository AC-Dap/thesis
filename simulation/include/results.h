#ifndef RESULTS_H
#define RESULTS_H

#include <map>
#include <string>

/**
* The results of our simulations are stored in a CSV file with 4 columns:
* - sketch_type: a string describing the kind of sketch used, including all parameters
* - k: the space used
* - n_trial: what number trial this row is for
* - estimate: the estimate of the moment
*/
struct Results {
    std::map<std::string,
      std::map<size_t,
          std::map<size_t, double>>> data;
    std::string output_path;

    Results(const std::string& output_path): output_path(output_path) {}

    bool has(const std::string& sketch_type, size_t k, size_t n_trial) {
        return data.find(sketch_type) != data.end() &&
               data[sketch_type].find(k) != data[sketch_type].end() &&
               data[sketch_type][k].find(n_trial) != data[sketch_type][k].end();
    }

    void add_result(const std::string& sketch_type, size_t k, size_t n_trial, double estimate) {
        data[sketch_type][k][n_trial] = estimate;
    }

    static Results read_from_file(const std::string& file_name);

    /**
     * Write the current contents of the results to the file.
     * This overwrites the file if it already exists.
     */
    void flush_to_file();
};

#endif //RESULTS_H
